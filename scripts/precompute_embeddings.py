"""Encode every target and peptide once with frozen ESM-2."""
import sys, time; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from peptide_eval import embed

cohort = pd.read_parquet("data/processed/cohort_split.parquet")
for kind, column, batch in (("targets", "target", 4), ("peptides", "peptide", 64)):
    sequences = sorted(cohort[column].unique())
    start = time.time()
    vectors = embed.cached(sequences, Path(f"data/processed/{kind}.npz"), batch_size=batch)
    print(f"{kind}: {vectors.shape} in {time.time()-start:.1f}s on {embed.device()}", flush=True)
