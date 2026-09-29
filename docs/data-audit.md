# Data audit: PPIKB

Source: `Affinity Dataset(main).xlsx` from the Duan Lab PPIKB release.
`sha256 0f31b10949195bf25aa258e7b14706f69d82859d8136319118bd81ffcb6ee4e2`, 7,786,520 bytes.

Every number below is produced by `scripts/control_check.py` from the raw file.
Re-run it to reproduce the table; nothing here is transcribed by hand.

## What the filters remove

| step | rows |
| --- | ---: |
| in the release | 21,845 |
| affinity is Kd, Ki or IC50 | 21,443 |
| linear backbone | 16,877 |
| no modified residues | 14,990 |
| standard 20 residues, peptide and target | 13,338 |
| usable numeric value | 13,338 |
| unique (target, peptide) after geometric-mean dedup | 9,078 |
| **targets with >= 10 peptides** | **203 targets, 5,516 pairs** |

The cyclic and modified peptides are dropped because ESM-2 cannot represent
them: it has one token per standard amino acid and nothing for a cyclisation
bond, so a cyclic peptide passed through it is silently read as a different,
linear molecule. This is a limit of the encoder, not of the data.

Kd, Ki and IC50 are pooled, which the published benchmarks also do. They are not
the same quantity, so this puts a ceiling on achievable accuracy that belongs to
the dataset rather than to any model.

## Why pooled correlation cannot be the headline metric

44.5% of the affinity variance in the cohort lies *between* targets rather than
within them. So a model that predicts each target's average affinity and ignores
the peptide entirely already explains nearly half the signal.

Measured on the 51 held-out targets, against an oracle given each target's own
mean and therefore structurally incapable of ranking peptides:

| control | pooled | per-target | enrichment | skill |
| --- | ---: | ---: | ---: | ---: |
| peptide-blind oracle (target mean) | **+0.678** | **+0.000** | 1.00x | -0.055 |
| peptide length alone | **+0.320** | +0.052 | 1.19x | -10.449 |
| train mean (deployable constant) | undefined | +0.000 | 1.00x | -0.778 |
| random | -0.046 | -0.052 | 0.97x | -6.317 |
| perfect (ceiling) | +1.000 | +1.000 | 4.47x | +1.000 |

The published leave-target-out result for this task is a pooled Spearman of
0.530 (arXiv 2608.30175). A model that cannot rank peptides at all scores 0.678
on that metric. The published number therefore sits *below* the peptide-blind
ceiling and cannot be read as evidence of within-target ranking ability.

The oracle is an upper bound, not an achievable score: a deployed model does not
know a held-out target's mean and would have to predict it from the target
sequence. The point is only that the metric does not separate the two abilities.

## Decisions

**Within-target ranking evaluation: GO.** 203 targets carry >= 10 peptides, all
203 have more than one distinct affinity value, and the median within-target
spread is about 4 orders of magnitude. Per-target Spearman is defined everywhere
it is needed.

**Probability calibration for screen hit-rate: NO-GO.** PPIKB collects published
binders and has no unbiased negative screen. Any hit-rate calibrated against
synthetic negatives would report the synthetic ratio.

**Scale calibration: GO, and it replaces the above.** The labels are continuous
affinities, so the calibration question is not "what fraction bind" but "is the
predicted affinity on the right part of the scale for an unseen target". That is
measurable as absolute error against the target's own median, reported as skill,
and it is the thing that actually breaks on a new target. See
`peptide_eval.metrics.per_target_skill`.
