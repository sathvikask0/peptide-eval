"""Bootstrap over held-out targets: is any model's ranking better than zero, or than another's?"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from peptide_eval import metrics, models

cohort = pd.read_parquet("data/processed/cohort_split.parquet")
tv = np.load("data/processed/targets.npz", allow_pickle=True)
pv = np.load("data/processed/peptides.npz", allow_pickle=True)
tix = {s: i for i, s in enumerate(tv["sequences"])}; pix = {s: i for i, s in enumerate(pv["sequences"])}
M = lambda f: (tv["vectors"][[tix[s] for s in f.target]], pv["vectors"][[pix[s] for s in f.peptide]])
train, test = cohort[cohort.split=="train"], cohort[cohort.split=="test"]
Xt_tr, Xp_tr = M(train); Xt_te, Xp_te = M(test)

per_target = {}
for kind in ("target_only","peptide_only","concat","interaction"):
    m = models.RidgeModel(kind).fit(Xt_tr, Xp_tr, train.affinity.values)
    per_target[kind] = metrics.per_target_spearman(
        test.target.values, test.affinity.values, m.predict(Xt_te, Xp_te))["per_target"]
per_target["peptide length"] = metrics.per_target_spearman(
    test.target.values, test.affinity.values, test.peptide.str.len().values.astype(float))["per_target"]

names = sorted(set.intersection(*(set(d) for d in per_target.values())))
table = pd.DataFrame({k: [v[n] for n in names] for k, v in per_target.items()}, index=names)
rng = np.random.default_rng(42)
draws = rng.integers(0, len(table), size=(2000, len(table)))
boot = np.stack([table.values[d].mean(0) for d in draws])

print(f"per-target Spearman over {len(table)} held-out targets, 2000 bootstrap resamples\n")
print(f"{'model':20s} {'mean':>7s} {'95% CI':>18s}  {'targets>0':>9s}")
for i, col in enumerate(table.columns):
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    print(f"{col:20s} {table[col].mean():+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {(table[col]>0).mean():8.0%}")

print("\npaired differences (same resampled targets):")
for a, b in [("interaction","peptide_only"), ("concat","peptide_only"), ("peptide_only","peptide length")]:
    d = boot[:, table.columns.get_loc(a)] - boot[:, table.columns.get_loc(b)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "differ" if (lo > 0) or (hi < 0) else "indistinguishable"
    print(f"  {a:14s} - {b:14s} = {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")
