"""Does the pooled metric reward a model that cannot rank peptides at all?"""
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from peptide_eval import data, metrics

frame, counts = data.load(Path("data/raw/ppikb-main.xlsx"))
pairs = data.deduplicate(frame)
keep = data.eligible_targets(pairs, minimum=10)
cohort = pairs[pairs.target.isin(keep)].reset_index(drop=True)

print("filter trace")
for k, v in counts.items(): print(f"  {k:26s} {v:6d}")
print(f"  {'unique pairs after dedup':26s} {len(pairs):6d}")
print(f"  {'targets with >=10 peptides':26s} {len(keep):6d}")
print(f"  {'pairs in cohort':26s} {len(cohort):6d}\n")

cohort.to_parquet("data/processed/cohort.parquet")

t, y = cohort.target.values, cohort.affinity.values
rng = np.random.default_rng(42)
controls = {
    "peptide-blind oracle (target mean)": cohort.groupby("target").affinity.transform("mean").values,
    "global mean (knows nothing)":        np.full(len(y), y.mean()),
    "random":                             rng.normal(size=len(y)),
    "perfect (upper bound)":              y,
}
print(f"{'control':36s} {'pooled':>8s} {'per-target':>11s} {'skill':>8s}")
for name, p in controls.items():
    r = metrics.report(t, y, p)
    ms = "  n/a" if r["macro_spearman"] is None else f"{r['macro_spearman']:+.3f}"
    print(f"{name:36s} {r['pooled_spearman']:+8.3f} {ms:>11s} {r['macro_skill']:+8.3f}")
