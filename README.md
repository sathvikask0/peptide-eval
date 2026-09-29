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

Measured against an oracle that is handed each held-out target's own mean
affinity, and is therefore structurally incapable of ranking peptides:

| control | pooled Spearman | per-target Spearman | skill |
| --- | ---: | ---: | ---: |
| peptide-blind oracle (target mean) | **+0.663** | **+0.000** | -0.046 |
| global mean | undefined | +0.000 | -0.862 |
| random | +0.011 | -0.004 | -6.079 |
| perfect | +1.000 | +1.000 | +1.000 |

The published leave-target-out result for this task is a pooled Spearman of
0.530 ([arXiv 2608.30175](https://arxiv.org/abs/2608.30175)). A model that
cannot rank peptides at all scores 0.663 on that metric, so the published number
sits below the peptide-blind ceiling and is not evidence of peptide-ranking
ability.

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

## Status

Controls and metrics are done and reproducible. Homology-clustered splits are
written but not yet run. Trained baselines (ESM-2 embeddings plus a regressor)
and a PepPrCLIP evaluation are next; the PepPrCLIP checkpoint is gated and the
access request is pending.

## Limits

Kd, Ki and IC50 are pooled, as the published benchmarks do. They are not the
same quantity, which puts a ceiling on achievable accuracy that belongs to the
dataset rather than to any model.

PPIKB collects published binders and has no unbiased negative screen, so nothing
here can estimate a prospective screening hit rate. Calibration is measured on
the affinity scale instead, which is defined without negatives.
