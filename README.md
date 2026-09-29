# peptide-eval

Can a peptide-binding model rank peptides for a target it has never seen?

Published benchmarks answer this with a correlation computed over all held-out
pairs at once. That number does not separate two very different abilities:
knowing which *targets* bind tightly in general, and knowing which *peptides*
suit a particular target. Only the second one is useful when you have one target
and a shortlist to narrow.

## The result

44.5% of the affinity variance in this dataset lies between targets rather than
within them. So a model can score well on a pooled correlation while being
unable to order peptides at all.

Measured on 51 held-out targets in 38 homology clusters, against an oracle that
is handed each target's own mean affinity and is therefore structurally
incapable of ranking peptides:

| control | pooled | per-target | enrichment | skill |
| --- | ---: | ---: | ---: | ---: |
| peptide-blind oracle (target mean) | **+0.678** | **+0.000** | 1.00x | -0.055 |
| peptide length alone | **+0.320** | +0.052 | 1.19x | -10.449 |
| train mean (deployable constant) | undefined | +0.000 | 1.00x | -0.778 |
| random | -0.046 | -0.052 | 0.97x | -6.317 |
| perfect (ceiling) | +1.000 | +1.000 | 4.47x | +1.000 |

The published leave-target-out result for this task is a pooled Spearman of
0.530 ([arXiv 2608.30175](https://arxiv.org/abs/2608.30175)). A model that
cannot rank peptides at all scores 0.663 on that metric, so the published number
sits below the peptide-blind ceiling and is not evidence of peptide-ranking
ability.

Peptide length is the sharper result. It is one integer per peptide, carries no
information about the target at all, and cannot express a preference between two
peptides of equal length. It scores +0.320 pooled.

The oracle is an upper bound rather than an achievable score: a deployed model
does not know a held-out target's mean and would have to predict it from the
target sequence. The claim here is about the metric, not about that paper's
models.

## Data

PPIKB, filtered to what a protein language model can actually read. 21,845 rows
become 203 targets carrying 5,516 peptide pairs. Every filter and its cost is in
[docs/data-audit.md](docs/data-audit.md); the numbers there are produced by
`scripts/control_check.py` from the raw file, not transcribed.

Raw data is not committed. Fetch `Affinity Dataset(main).xlsx` from the PPIKB
release into `data/raw/` (the audit records its checksum) and run:

```
uv run --with pandas --with openpyxl --with numpy --with scipy --with pyarrow \
  python scripts/control_check.py
```

## What this measures

- `metrics.per_target_spearman` — rank peptides inside each held-out target,
  then average. A prediction that is constant within a target scores 0.
- `metrics.per_target_skill` — absolute error against the target's own median
  affinity. Ranking is invariant to monotone rescaling, so a model can order
  peptides perfectly and still put every prediction on the wrong part of the
  scale. This is the half that breaks on an unseen target.
- `metrics.pooled_spearman` — kept only to show the gap between it and the
  per-target number.
- `cluster` — single-linkage homology clusters, so a split can separate protein
  families rather than exact sequences. Published leave-target-out splits
  exclude by exact sequence only, which lets a near-copy of a test target sit in
  training.

## Splits

203 targets fall into 101 homology clusters by local-alignment identity with a
coverage requirement. Whole clusters go to train / validation / test at
122 / 30 / 51 targets. No test-train pair meets the relatedness rule; the
closest surviving pair is 46.6% identity over 49.2% coverage.

Percent identity alone does not separate related proteins from unrelated ones
here: random target pairs have a median best-local-alignment identity of 0.33.
What separates them is coverage, at a median of 0.08. Requiring both calls 0.8%
of random pairs related while an identical sequence and a 20%-mutated copy both
pass.

## Status

Controls, metrics and homology-clustered splits are done and reproducible.
Trained baselines (ESM-2 embeddings plus a regressor) are next. A PepPrCLIP
evaluation follows if the gated checkpoint is approved; nothing above depends
on it.

## Limits

Kd, Ki and IC50 are pooled, as the published benchmarks do. They are not the
same quantity, which puts a ceiling on achievable accuracy that belongs to the
dataset rather than to any model.

PPIKB collects published binders and has no unbiased negative screen, so nothing
here can estimate a prospective screening hit rate. Calibration is measured on
the affinity scale instead, which is defined without negatives.
