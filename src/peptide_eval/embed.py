"""Frozen ESM-2 representations for targets and peptides.

The encoder is never trained here. Everything downstream is a model fitted on
these vectors, which keeps the comparison between models about the model rather
than about how long each one was allowed to fine-tune.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MODEL = "facebook/esm2_t33_650M_UR50D"

# ESM-2 was trained with 1024 positions, two of which the tokeniser spends on
# the start and end tokens.
WINDOW = 1022


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _windows(sequence: str) -> list[str]:
    """Split a sequence too long for the context into consecutive windows.

    37 of the 203 targets are longer than the window, one of them by a factor of
    seven. Truncating them would silently discard most of the protein, and the
    binding site is not guaranteed to be in the first 1,022 residues. Chunking
    and pooling keeps every residue, at the cost of no window seeing the
    long-range contacts that cross its boundary.
    """
    return [sequence[i : i + WINDOW] for i in range(0, len(sequence), WINDOW)] or [sequence]


@torch.no_grad()
def encode(sequences: list[str], batch_size: int = 8) -> np.ndarray:
    """Mean-pooled residue representations, one row per input sequence.

    Mean pooling over residues rather than the start token: ESM-2 is a masked
    language model with no sentence-level training objective, so its start token
    was never trained to summarise the sequence.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device()).eval()

    # Every window of every sequence is encoded in one flat pass, then windows
    # are pooled back into their sequence weighted by how many residues each
    # holds, so a 7,073-residue protein is not dominated by its short tail.
    flat, owner, weight = [], [], []
    for index, sequence in enumerate(sequences):
        for window in _windows(sequence):
            flat.append(window)
            owner.append(index)
            weight.append(len(window))

    vectors = []
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device())
        states = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(pooled.float().cpu().numpy())

    vectors = np.concatenate(vectors)
    owner, weight = np.array(owner), np.array(weight, dtype=np.float32)
    out = np.zeros((len(sequences), vectors.shape[1]), dtype=np.float32)
    totals = np.zeros(len(sequences), dtype=np.float32)
    np.add.at(out, owner, vectors * weight[:, None])
    np.add.at(totals, owner, weight)
    return out / totals[:, None]


def cached(sequences: list[str], path: Path, batch_size: int = 8) -> np.ndarray:
    """Encode once and reuse. Keyed by the sequence list, so a changed cohort re-encodes."""
    if path.exists():
        store = np.load(path, allow_pickle=True)
        if list(store["sequences"]) == list(sequences):
            return store["vectors"]
    vectors = encode(sequences, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sequences=np.array(sequences, dtype=object), vectors=vectors)
    return vectors
