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

## What the models do

Ridge probes on frozen ESM-2 650M embeddings, fitted on 122 training targets and
scored on the 51 held-out ones. Each model differs only in what it is allowed to
see.

| model | pooled | per-target | 95% CI | enrichment |
| --- | ---: | ---: | :---: | ---: |
| target only | +0.102 | +0.000 | [+0.000, +0.000] | 1.00x |
| peptide only | +0.230 | +0.095 | [-0.009, +0.201] | 1.46x |
| target + peptide | +0.148 | +0.093 | [-0.011, +0.191] | 1.47x |
| target x peptide | +0.187 | +0.071 | [-0.040, +0.173] | 1.24x |

Every interval crosses zero. On 2,000 bootstrap resamples over targets, no model
ranks peptides within an unseen target measurably better than chance, and no
model is distinguishable from any other. The model that sees both target and
peptide is not distinguishable from the one that sees only the peptide, and
ESM-2's peptide embedding is not distinguishable from the peptide's length.

Read carefully, this says we could not detect target-conditioned ranking, not
that none exists. 51 held-out targets give a confidence interval about 0.10 wide,
so an effect of the size these models show would need roughly 65 targets to
separate from zero. Lowering the 10-peptide cutoff would buy those targets at
the cost of noisier per-target correlations.

## Is it the probe or the representation?

A random forest on the same features, same splits, averaged over ten forest
seeds:

| model | ridge | forest | forest sd |
| --- | ---: | ---: | ---: |
| target only | +0.000 | +0.000 | 0.000 |
| peptide only | +0.095 | **+0.102** | 0.015 |
| target + peptide | +0.093 | +0.085 | 0.025 |
| target x peptide | +0.071 | +0.072 | 0.034 |

Swapping a linear probe for a forest changes nothing: peptide-only moves from
+0.095 to +0.102, inside the seed noise. Train R2 meanwhile rises from 0.44 to
0.87, so the forest fits the training pairs far harder and generalises
identically. The limit is the frozen representation, not the probe.

The ordering is the result worth keeping. The more target information a model is
given, the lower its mean and the higher its variance across seeds. Paired over
seeds, the interaction model is 0.030 *behind* peptide-only and ahead on 3 of 10.

A single forest seed is not enough to see this. On seed 42 the interaction model
scores +0.110 with a bootstrap interval of [+0.017, +0.209], which excludes zero
and would have been reported as the first positive result. Ten seeds put that
value near the top of its own noise distribution. Seed variation is checked here
for that reason.

Trees are not additive, so a forest on concatenated features is target-
conditioned where ridge is not: the rank correlation between two targets'
orderings is 0.82 rather than 1.000000. It still does not beat peptide-only.

Neither model class has been fine-tuned, and no model here uses cross-attention
between target and peptide residues rather than pooled vectors. Either could do
better, and should be run against these same controls.

Two structural checks hold exactly, which is the evidence that the harness
measures what it claims. The target-only model scores per-target +0.000, because
its prediction cannot vary within a target. And a linear model on concatenated
features predicts `w_t.x_t + w_p.x_p`, so for a fixed target the first term is a
constant offset and the peptide ordering is the same for every target: measured
across two different targets, the rank correlation between its orderings is
1.000000. The interaction model, which has an `x_t * x_p` term, gives 0.376.

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

The benchmark, the controls, the splits and the ESM-2 ridge baselines are done
and reproducible end to end. A PepPrCLIP evaluation follows if the gated
checkpoint is approved; nothing above depends on it.

## Limits

Kd, Ki and IC50 are pooled, as the published benchmarks do. They are not the
same quantity, which puts a ceiling on achievable accuracy that belongs to the
dataset rather than to any model.

PPIKB collects published binders and has no unbiased negative screen, so nothing
here can estimate a prospective screening hit rate. Calibration is measured on
the affinity scale instead, which is defined without negatives.
