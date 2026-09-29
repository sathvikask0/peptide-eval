"""Fit ridge probes on frozen ESM-2 features and score them against the controls."""
import sys, json; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from peptide_eval import metrics, models

cohort = pd.read_parquet("data/processed/cohort_split.parquet")
tv = np.load("data/processed/targets.npz", allow_pickle=True)
pv = np.load("data/processed/peptides.npz", allow_pickle=True)
tix = {s: i for i, s in enumerate(tv["sequences"])}
pix = {s: i for i, s in enumerate(pv["sequences"])}

def matrices(frame):
    return (tv["vectors"][[tix[s] for s in frame.target]],
            pv["vectors"][[pix[s] for s in frame.peptide]],
            frame.affinity.values, frame.target.values)

train, test = cohort[cohort.split == "train"], cohort[cohort.split == "test"]
Xt_tr, Xp_tr, y_tr, _ = matrices(train)
Xt_te, Xp_te, y_te, t_te = matrices(test)
print(f"train {len(train)} pairs / {train.target.nunique()} targets | "
      f"test {len(test)} pairs / {test.target.nunique()} targets")
print(f"embedding dim {Xt_tr.shape[1]}\n")

rng = np.random.default_rng(42)
rows = []
def score(name, pred):
    r = metrics.report(t_te, y_te, pred)
    rows.append({"model": name, **r})
    f = lambda v: "  n/a" if v is None else f"{v:+.3f}"
    print(f"{name:24s} {f(r['pooled_spearman']):>8s} {f(r['macro_spearman']):>11s} "
          f"{r['macro_enrichment']:8.2f} {f(r['macro_skill']):>9s}")

print(f"{'':24s} {'pooled':>8s} {'per-target':>11s} {'enrich':>8s} {'skill':>9s}")
score("random", rng.normal(size=len(y_te)))
score("peptide length", test.peptide.str.len().values.astype(float))
score("peptide-blind oracle", test.groupby("target").affinity.transform("mean").values)
print()
for kind in ("target_only", "peptide_only", "concat", "interaction"):
    model = models.RidgeModel(kind).fit(Xt_tr, Xp_tr, y_tr)
    score(f"esm2 ridge {kind}", model.predict(Xt_te, Xp_te))

json.dump(rows, open("results/baselines_test.json", "w"), indent=2, default=float)

# A linear model on concatenated features predicts w_t.x_t + w_p.x_p, so for a
# fixed target the first term is a constant offset and the ordering of peptides
# is w_p.x_p regardless of which target it is. Concatenation therefore applies
# one global peptide ranking to every target: it can learn that some peptides
# bind widely, never that a peptide suits this target in particular. The
# interaction model has an x_t * x_p term and is not bound by this.
#
# The test: score the same peptide set against two different targets and
# correlate the two orderings. Concat must give exactly 1.0.
from scipy.stats import spearmanr
probe = pv["vectors"][[pix[s] for s in sorted(test.peptide.unique())[:200]]]
print("\nstructural check: is one global peptide ranking reused for every target?")
for kind in ("concat", "interaction"):
    model = models.RidgeModel(kind).fit(Xt_tr, Xp_tr, y_tr)
    a, b = tv["vectors"][[tix[test.target.iloc[0]]]], tv["vectors"][[tix[test.target.iloc[-1]]]]
    first = model.predict(np.repeat(a, len(probe), 0), probe)
    second = model.predict(np.repeat(b, len(probe), 0), probe)
    rho = spearmanr(first, second).statistic
    verdict = "one global ranking" if abs(rho - 1) < 1e-9 else "target-conditioned"
    print(f"  {kind:12s} rho between two targets' orderings = {rho:.6f}  -> {verdict}")
