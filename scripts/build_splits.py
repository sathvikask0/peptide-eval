"""Cluster eligible targets by homology and assign whole clusters to splits."""
import sys, json, time; sys.path.insert(0, "src")
import pandas as pd
from peptide_eval import cluster

cohort = pd.read_parquet("data/processed/cohort.parquet")
targets = sorted(cohort.target.unique())
print(f"{len(targets)} targets; lengths {min(map(len,targets))}-{max(map(len,targets))} aa", flush=True)

start = time.time()
assignment = cluster.cluster(targets)
sizes = pd.Series(list(assignment.values())).value_counts()
print(f"clustered in {time.time()-start:.1f}s -> {len(sizes)} clusters")
print(f"  singletons {(sizes==1).sum()}   largest {sizes.max()}   "
      f"sizes {dict(sizes.value_counts().sort_index())}\n")

split = cluster.split_by_cluster(assignment, seed=42)
cohort["cluster"] = cohort.target.map(assignment)
cohort["split"] = cohort.target.map(split)
cohort.to_parquet("data/processed/cohort_split.parquet")

print("split            targets  clusters   pairs")
for name in ("train", "validation", "test"):
    s = cohort[cohort.split == name]
    print(f"{name:14s} {s.target.nunique():8d} {s.cluster.nunique():9d} {len(s):7d}")

# A split is only honest if no test target is related to a training target.
aligner = cluster._aligner()
train = sorted(cohort[cohort.split == "train"].target.unique())
test = sorted(cohort[cohort.split == "test"].target.unique())
violations, worst = 0, (0.0, 0.0)
for a in test:
    ka = cluster._kmers(a)
    for b in train:
        kb = cluster._kmers(b)
        if len(ka & kb) / max(1, min(len(ka), len(kb))) < cluster.PREFILTER:
            continue
        identity, coverage = cluster.similarity(a, b, aligner)
        if identity >= cluster.MIN_IDENTITY and coverage >= cluster.MIN_COVERAGE:
            violations += 1
        if identity * coverage > worst[0] * worst[1]:
            worst = (identity, coverage)
print(f"\nleakage check over {len(test)}x{len(train)} test-train pairs")
print(f"  pairs meeting the relatedness rule: {violations}  (must be 0)")
print(f"  closest surviving pair: identity {worst[0]:.3f}, coverage {worst[1]:.3f}")

json.dump({"targets": len(targets), "clusters": int(len(sizes)),
           "splits": {n: int(cohort[cohort.split==n].target.nunique())
                      for n in ("train","validation","test")},
           "leakage_violations": violations},
          open("data/processed/split_summary.json", "w"), indent=2)
