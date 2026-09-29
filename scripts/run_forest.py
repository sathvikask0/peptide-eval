"""Random forest on frozen ESM-2 features, same splits and controls as the ridge probes."""
import sys, json, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from peptide_eval import metrics, models

cohort = pd.read_parquet("data/processed/cohort_split.parquet")
tv = np.load("data/processed/targets.npz", allow_pickle=True)
pv = np.load("data/processed/peptides.npz", allow_pickle=True)
tix = {s: i for i, s in enumerate(tv["sequences"])}; pix = {s: i for i, s in enumerate(pv["sequences"])}
M = lambda f: (tv["vectors"][[tix[s] for s in f.target]], pv["vectors"][[pix[s] for s in f.peptide]])
train, test = cohort[cohort.split == "train"], cohort[cohort.split == "test"]
Xt_tr, Xp_tr = M(train); Xt_te, Xp_te = M(test)
y_tr, y_te, t_te = train.affinity.values, test.affinity.values, test.target.values
print(f"train {len(train)} pairs / {train.target.nunique()} targets | "
      f"test {len(test)} pairs / {test.target.nunique()} targets\n")

rows, per_target = [], {}
print(f"{'model':26s} {'pooled':>8s} {'per-target':>11s} {'enrich':>8s} {'skill':>9s} {'trainR2':>8s} {'sec':>6s}")
for kind in ("target_only", "peptide_only", "concat", "interaction"):
    start = time.time()
    model = models.ForestModel(kind).fit(Xt_tr, Xp_tr, y_tr)
    pred = model.predict(Xt_te, Xp_te)
    r = metrics.report(t_te, y_te, pred)
    per_target[kind] = metrics.per_target_spearman(t_te, y_te, pred)["per_target"]
    rows.append({"model": f"forest {kind}", **r})
    f = lambda v: "  n/a" if v is None else f"{v:+.3f}"
    print(f"forest {kind:19s} {f(r['pooled_spearman']):>8s} {f(r['macro_spearman']):>11s} "
          f"{r['macro_enrichment']:8.2f} {f(r['macro_skill']):>9s} "
          f"{model.forest.score(models.features(kind,Xt_tr,Xp_tr),y_tr):8.3f} {time.time()-start:6.0f}")
json.dump(rows, open("results/forest_test.json", "w"), indent=2, default=float)

names = sorted(set.intersection(*(set(d) for d in per_target.values())))
table = pd.DataFrame({k: [v[n] for n in names] for k, v in per_target.items()}, index=names)
rng = np.random.default_rng(42)
boot = np.stack([table.values[d].mean(0) for d in rng.integers(0, len(table), size=(2000, len(table)))])
print(f"\nbootstrap over {len(table)} held-out targets")
for i, col in enumerate(table.columns):
    lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
    print(f"  {col:14s} {table[col].mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"{'   excludes zero' if lo > 0 or hi < 0 else ''}")
for a, b in [("interaction","peptide_only"), ("concat","peptide_only")]:
    d = boot[:, table.columns.get_loc(a)] - boot[:, table.columns.get_loc(b)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {a} - {b} = {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"  {'differ' if lo > 0 or hi < 0 else 'indistinguishable'}")

# Trees are not additive, so concat is target-conditioned here. Verify rather than assume.
probe = pv["vectors"][[pix[s] for s in sorted(test.peptide.unique())[:200]]]
print("\nis one global peptide ranking reused for every target?")
for kind in ("concat", "interaction"):
    m = models.ForestModel(kind).fit(Xt_tr, Xp_tr, y_tr)
    a, b = tv["vectors"][[tix[test.target.iloc[0]]]], tv["vectors"][[tix[test.target.iloc[-1]]]]
    rho = spearmanr(m.predict(np.repeat(a, len(probe), 0), probe),
                    m.predict(np.repeat(b, len(probe), 0), probe)).statistic
    print(f"  {kind:12s} rho between two targets' orderings = {rho:.6f}"
          f"  -> {'one global ranking' if abs(rho-1) < 1e-9 else 'target-conditioned'}")
