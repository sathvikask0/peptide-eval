import sys, time; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from peptide_eval import residues, embed
cohort = pd.read_parquet("data/processed/cohort_split.parquet")
for kind, column, batch in (("targets", "target", 2), ("peptides", "peptide", 64)):
    seqs = sorted(cohort[column].unique())
    t = time.time()
    v, o = residues.cached(seqs, Path(f"data/processed/{kind}_residues.npz"), batch_size=batch)
    print(f"{kind}: {v.shape} residues, {len(seqs)} sequences, "
          f"{v.nbytes/1e6:.0f} MB, {time.time()-t:.0f}s on {embed.device()}", flush=True)
