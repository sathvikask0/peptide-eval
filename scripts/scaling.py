"""Does the target signal grow with the number of training targets, or is it absent?

Raw score is the wrong axis: most of it comes from the peptide path, which does
not care how many targets the model saw. The gap between scoring a test target
with its own residues and with another target's residues is target information
and nothing else, so that is what is plotted against training-set size.
"""
import sys, json; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
sys.argv = [sys.argv[0]]
exec(open("scripts/run_attention.py").read().split("test = cohort")[0])

test = cohort[cohort.split == "test"]
names = sorted(test.target.unique())
swap = dict(zip(names, np.random.default_rng(0).permutation(names)))
train_targets = sorted(cohort[cohort.split == "train"].target.unique())

def train_on(subset, seed):
    """Same trainer, restricted to a subset of training targets."""
    global cohort
    keep = cohort[(cohort.split != "train") | (cohort.target.isin(subset))]
    saved, cohort = cohort, keep
    try:
        return train(True, seed)[0]
    finally:
        cohort = saved

rows = []
print(f"{'targets':>8s} {'pairs':>6s} {'seed':>5s} {'correct':>8s} {'wrong':>7s} {'gap':>7s}")
for n in (30, 60, 90, 122):
    for seed in (0, 1, 2, 3, 4):
        subset = list(np.random.default_rng(100 + seed).permutation(train_targets)[:n])
        pairs = int(cohort[(cohort.split=="train") & (cohort.target.isin(subset))].shape[0])
        model = train_on(subset, seed)
        right = score(test, predict(model, test))
        wrong = score(test, predict(model, test, swap=swap))
        rows.append({"targets": n, "pairs": pairs, "seed": seed,
                     "correct": right, "wrong": wrong, "gap": right - wrong})
        print(f"{n:8d} {pairs:6d} {seed:5d} {right:+8.3f} {wrong:+7.3f} {right-wrong:+7.3f}")

json.dump(rows, open("results/scaling.json", "w"), indent=2, default=float)
frame = pd.DataFrame(rows)
print(f"\n{'targets':>8s} {'pairs':>6s} {'correct':>16s} {'gap (target signal)':>22s}")
for n, g in frame.groupby("targets"):
    print(f"{n:8d} {int(g.pairs.mean()):6d} {g.correct.mean():+10.3f} +- {g.correct.std():.3f}"
          f" {g.gap.mean():+14.3f} +- {g.gap.std():.3f}")
lo = frame[frame.targets == 30].gap
hi = frame[frame.targets == 122].gap
print(f"\ngap at 30 targets {lo.mean():+.3f}, at 122 targets {hi.mean():+.3f}, "
      f"change {hi.mean()-lo.mean():+.3f} (seed sd ~{frame.gap.std():.3f})")
