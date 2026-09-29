"""Score the controls on the held-out test clusters."""
import sys, json; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from peptide_eval import metrics

cohort = pd.read_parquet("data/processed/cohort_split.parquet")
test = cohort[cohort.split == "test"].reset_index(drop=True)
train = cohort[cohort.split == "train"]
print(f"test: {test.target.nunique()} targets, {test.cluster.nunique()} clusters, {len(test)} pairs\n")

t, y = test.target.values, test.affinity.values
rng = np.random.default_rng(42)
controls = {
    "peptide-blind oracle": test.groupby("target").affinity.transform("mean").values,
    "train mean (deployable)": np.full(len(y), train.affinity.mean()),
    "random": rng.normal(size=len(y)),
    "peptide length": test.peptide.str.len().values.astype(float),
    "perfect (ceiling)": y,
}
rows = []
print(f"{'control':26s} {'pooled':>8s} {'per-target':>11s} {'enrich':>8s} {'skill':>8s}")
for name, p in controls.items():
    r = metrics.report(t, y, p)
    fmt = lambda v: " n/a" if v is None else f"{v:+.3f}"
    print(f"{name:26s} {fmt(r["pooled_spearman"]):>8s} {fmt(r['macro_spearman']):>11s} "
          f"{r['macro_enrichment']:8.2f} {fmt(r['macro_skill']):>8s}")
    rows.append({"control": name, **r})
json.dump(rows, open("results/controls_test.json", "w"), indent=2, default=float)
print(f"\ntargets scored: {rows[0]['targets_scored']} of {test.target.nunique()}")
