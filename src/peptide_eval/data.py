"""Build the canonical evaluation dataset from the PPIKB release.

Every filter here is a choice that changes what the benchmark measures, so each
one states what it removes and why rather than being folded into one expression.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SHEET = "ppi_research"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# PPIKB stores the same measurement in uM, nM and pM columns. nM is the one that
# is populated for every row, so it is the single source and the others are
# ignored rather than reconciled.
VALUE_COLUMN = "nM"
TARGET_COLUMN = "Protein_Sequence"
PEPTIDE_COLUMN = "Peptide_Sequence"

# Kd, Ki and IC50 are pooled. They are not the same quantity: IC50 depends on
# assay conditions in a way Kd does not. Pooling them is what the published
# benchmarks do, and keeping it makes our numbers comparable to theirs, but it
# puts a ceiling on how well any model can do that is not the model's fault.
AFFINITY_KINDS = {"Kd", "KD", "Ki", "IC50"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_canonical(sequence: str) -> bool:
    """True when a protein language model can read the sequence unchanged.

    ESM-2 has one token per standard amino acid and nothing for a cyclisation
    bond or a modified residue. A cyclic peptide passed through it is silently
    read as its linear sequence, which is a different molecule.
    """
    return len(sequence) > 0 and set(sequence) <= STANDARD_AA


def load(path: Path) -> pd.DataFrame:
    """Read PPIKB and reduce it to (target, peptide, affinity) rows we can model."""
    raw = pd.read_excel(path, sheet_name=SHEET)
    counts = {"rows_in_release": len(raw)}

    frame = raw.copy()
    frame["peptide"] = frame[PEPTIDE_COLUMN].astype(str).str.strip().str.upper()
    frame["target"] = frame[TARGET_COLUMN].astype(str).str.strip().str.upper()

    frame = frame[frame["Affinity_Parameters"].isin(AFFINITY_KINDS)]
    counts["after_affinity_kind"] = len(frame)

    frame = frame[frame["Linear/Cyclic"].eq("Linear")]
    counts["after_linear_only"] = len(frame)

    frame = frame[frame["Residue_Modification"].eq("No")]
    counts["after_unmodified_only"] = len(frame)

    frame = frame[frame["peptide"].map(is_canonical) & frame["target"].map(is_canonical)]
    counts["after_standard_residues"] = len(frame)

    value = pd.to_numeric(frame[VALUE_COLUMN], errors="coerce")
    frame = frame[value > 0]
    # -log10 of molar concentration: bigger is tighter binding, and the log is
    # what makes an error of "2x" cost the same at 1 nM and at 1 uM.
    frame = frame.assign(affinity=-np.log10(value[value > 0] * 1e-9))
    counts["after_usable_value"] = len(frame)

    return frame[["target", "peptide", "affinity", "Affinity_Parameters", "UniProt_ID"]], counts


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeat measurements of one pair to their geometric mean.

    The mean is taken in log space, which is the geometric mean of the
    concentrations. PPIKB carries the same pair from several papers and those
    disagree by more than a factor of two, so keeping every row would weight a
    well-studied pair more heavily for no reason connected to binding.
    """
    return (
        frame.groupby(["target", "peptide"], as_index=False)
        .agg(affinity=("affinity", "mean"), measurements=("affinity", "size"))
    )


def eligible_targets(pairs: pd.DataFrame, minimum: int = 10) -> pd.Index:
    """Targets with enough peptides to rank.

    Ranking within a target is the thing we are measuring, so a target with
    three peptides contributes a Spearman correlation over three points, which
    is mostly noise. The cutoff buys reliability per target at the cost of
    fewer targets.
    """
    counts = pairs.groupby("target").size()
    return counts[counts >= minimum].index
