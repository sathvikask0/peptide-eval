# Protocol: evaluating peptide binders on unseen targets

Dataset: PPIKB, canonical linear subset. See [data-audit.md](data-audit.md).

Nothing here is locked. Each choice below says what it costs, so that changing
one is a decision rather than a correction.

## 1. The question

Does a sequence-based model rank candidate peptides usefully *within* a target
protein it has not seen?

Three things follow from the word "within".

**Metrics are computed per target and then averaged.** A correlation pooled over
all held-out pairs sums two abilities: knowing which targets bind tightly in
general, and knowing which peptides suit one target. In this dataset 44.5% of
affinity variance is between targets, so the pooled number is mostly the first
ability. A peptide-blind oracle scores pooled +0.663 and per-target +0.000.

**Held-out means the sequence family, not the sequence.** A two-residue variant
of a training target binds much the same peptides. Exact-sequence exclusion,
which the published benchmarks use, does not prevent that.

**Every model runs beside controls that cannot cheat.** A control that scores
well is a defect in the metric, and we would rather find it before the model
does.

## 2. Calibration

**Hit-rate calibration: NO-GO.** PPIKB collects published binders and has no
unbiased negative screen. A score-to-probability map fitted against synthetic
negatives reports the synthetic ratio, not a wet-lab hit rate, and no such
number will be claimed.

**Scale calibration: GO.** The labels are continuous affinities, so the question
that survives is whether a predicted affinity lands on the right part of the
scale for an unseen target. Measured as mean absolute error against that
target's own median, reported as skill: 0 means the model knows nothing beyond
the median, negative means it is confidently wrong about the scale.

This reference is generous — it uses the held-out target's own labels, which a
deployed model would not have. A model that cannot beat it has no scale
knowledge at all.

## 3. Dataset

Inclusion: standard 20 residues for target and peptide, linear backbone, no
modified residues, a usable Kd/Ki/IC50, and at least 10 distinct peptides per
target. Repeat measurements of one pair are collapsed by geometric mean, which
is the arithmetic mean in log space. Endpoint is `-log10(molar)`.

Target length is deliberately *not* an inclusion criterion. ESM-2's 1024-token
context is a property of one encoder, and excluding 19% of targets to suit it
would let the encoder define the benchmark. Long targets stay in; how each model
handles them is recorded with that model.

Kd, Ki and IC50 are pooled, as the published benchmarks do. They are not the
same quantity, and this ceiling belongs to the dataset rather than to any model.

## 4. Splits

Targets are clustered by sequence identity from a global pairwise alignment,
normalised by the shorter sequence so a domain contained in a larger protein
counts as related. A 3-mer overlap prefilter skips pairs that cannot clear the
threshold; it is an optimisation, not the similarity measure. Clusters are
single-linkage connected components at 30% identity — deliberately aggressive,
because the failure it prevents is testing on a near-copy.

Whole clusters go to train / validation / test at 60 / 15 / 25 by target count.
Test gets the largest share because the headline number is an average over
held-out targets, and its reliability is set by how many there are.

Every split run reports the highest identity between any test and any train
target. If that exceeds the clustering threshold, the split is broken.

**Pending:** cross-reference held-out targets against PepPrCLIP's training set,
to flag any target that model has structural precedent for. Needs the checkpoint,
which is gated.

## 5. Metrics

Per held-out target:

- **Spearman correlation** between predicted score and measured affinity. A
  prediction that is constant within a target orders nothing and scores 0, not
  undefined — failing to discriminate is a result, not missing data.
- **Top-20% enrichment**: of the tightest 20% of binders, the fraction the model
  places in its own top 20%, divided by the 0.2 that random selection gives.
  This is the number that maps onto a decision, because a screen tests a fixed
  number of wells.
- **Skill** against the target's own median, as in §2.

Aggregated as the mean over targets, with a 95% bootstrap interval over targets
(1,000 resamples), plus the share of targets with rho > 0.3, > 0.5, and <= 0.
Pooled Spearman is reported alongside, only to show the gap.

NDCG was considered and dropped: its discount function is an arbitrary choice
here, and it answers nothing that Spearman and enrichment do not.

## 6. Controls

Run under identical splits with every model.

1. **Random scores.** Expect rho 0, enrichment 1.0x.
2. **Global constant.** Predicts the training mean. Scores exactly 0 per target,
   by definition rather than by tie-breaking.
3. **Target-only.** A regressor on target embeddings alone. Its prediction is
   constant within a target, so per-target rho is 0 by construction while pooled
   rho stays high. This is the control that indicts the pooled metric.
4. **Peptide-only.** A regressor on peptide embeddings alone — how much ranking
   comes from peptides being generically sticky, independent of the target.
5. **Nearest neighbour.** Affinity of the most similar training pair. Tests
   whether anything beyond retrieval is needed.

Control 4 is the demanding one. A model only demonstrates target-conditioned
ranking if it beats the peptide-only regressor, since that baseline already
captures everything explainable by the peptide alone.

## 7. Execution

Local, Apple Silicon via `mps`. Target embeddings precomputed once. Seed 42 for
clustering, splitting and bootstrapping.
