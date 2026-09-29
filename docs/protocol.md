# Research Protocol: Evaluating Peptide Binders on Unseen Targets

**Status:** Locked after Data Feasibility Audit (2026-09-29)  
**Dataset:** PPIKB Main Canonical Linear Subset  
**Target Architecture:** Sequence-only contrastive protein-peptide models (PepPrCLIP / ESM-2 baselines)

---

## 1. Core Evaluation Questions

### Primary Question
Does a sequence-based peptide-protein model rank candidate peptides usefully within a previously unseen target protein?

### What Defines "Useful Ranking"?
1. **Target-Disjoint Evaluation:** The model must evaluate targets whose exact sequence (and sequence cluster at $\sim 30\%$ identity) was never seen during training or validation.
2. **Within-Target Discrimination:** Metrics must be calculated **per target** and then summarized across targets. A pooled correlation over all pairs conflates across-target baseline differences with true peptide prioritization.
3. **Control Comparison:** The model must outperform simple controls (constant prediction, target-only mean, peptide-only mean, and random shuffle).

### Calibration Policy
- **Probability Calibration is frozen as NO-GO for prospective yield claims.** PPIKB contains curated positive affinities from published literature. It lacks unbiased negative library screens. Any score-to-probability mapping derived from synthetic negatives would only reflect the synthetic negative ratio and will not be claimed as prospective wet-lab success probabilities.

---

## 2. Frozen Dataset & Preprocessing Specifications

1. **Inclusion Criteria:**
   - Standard 20 canonical amino acids for both target and peptide.
   - Linear peptide backbones (`Linear/Cyclic == 'Linear'`).
   - Target sequence length $\le 1024$ aa (matching native ESM-2 context).
   - Target must have $\ge 10$ distinct peptides with measured affinities.
2. **Deduplication Rule:**
   - Multiple experimental measurements for the identical `(target_sequence, peptide_sequence)` pair are aggregated using the **geometric mean** of their nanomolar ($nM$) values.
3. **Ground Truth Endpoint:**
   - $-\log_{10}(K_d \text{ or } IC_{50} \text{ in } M)$ (higher value = stronger binding).

---

## 3. Split & Homology Separation Strategy

- **Target Clustering:** Cluster all eligible target sequences using connected-component graph clustering at sequence identity threshold $T \approx 0.30$ (k-mer Jaccard distance).
- **Cluster-Disjoint Splits:**
  - 5-Fold Target-Cluster Cross-Validation (or 60% Train / 20% Val / 20% Test by cluster).
  - All measurements for all targets belonging to a cluster remain strictly within the same fold.
  - No sequence from a test cluster may appear in training or validation.
- **PDB Overlap Audit:** Cross-reference held-out test targets against known PDB co-crystal IDs to flag targets with structural training precedents in PepPrCLIP's training set.

---

## 4. Evaluation Metrics & Statistical Analysis

### Within-Target Metrics (Calculated for each held-out test target $i$):
1. **Spearman Rank Correlation ($\rho_i$):**
   $$\rho_i = \text{SpearmanCorr}(\hat{s}_{i, :}, y_{i, :})$$
   where $\hat{s}$ are model predicted scores and $y$ are true binding strengths.
2. **Top-20% Hit Enrichment ($E_{20, i}$):**
   Fraction of true top-20% tightest binders captured in the model's top-20% highest-ranked predictions relative to random selection.
3. **Normalized Discounted Cumulative Gain (NDCG@k):**
   Measures ranking quality with higher weight placed on correctly identifying the highest-affinity binders.

### Aggregate Reporting:
- **Macro-Averaged Spearman $\rho$:** Mean and median $\rho$ across all held-out targets.
- **Distribution & Violin Plots:** Showing per-target variability and percentage of targets with $\rho > 0.3$, $\rho > 0.5$, and $\rho \le 0$.
- **Resampling Uncertainty:** 95% bootstrap confidence intervals across targets (1,000 resamples).
- **Pooled Metrics (Secondary):** Reported for comparison with prior literature (e.g. Tian et al.), explicitly noting the confounding effect of target baseline variation.

---

## 5. Baselines and Negative Controls

Every model evaluation must be run alongside the following controls under identical split conditions:

1. **Random Control:** Uniform random score assignment (expectation: $\rho = 0.0$, Enrichment $= 1.0\times$).
2. **Constant Predictor:** Predicts the global training mean for all pairs (ranking undefined; breaks ties randomly).
3. **Target-Only Predictor:** Fits a regressor on target ESM-2 embeddings alone. Because predicted scores are identical for all peptides on a given target, within-target ranking is mathematically 0 / random, demonstrating that target-level affinity knowledge does not enable peptide ranking.
4. **Peptide-Only Predictor:** Fits a regressor on peptide ESM-2 embeddings alone (predicts general "sticky" peptides). Evaluates how much ranking performance is driven purely by peptide composition regardless of the target.
5. **K-Nearest Neighbors / Sequence Similarity:** Predicts affinity based on sequence similarity to observed training targets and peptides.

---

## 6. Execution Protocol

- All code and weights run locally on Apple Silicon (M3 Pro, 18 GB RAM) via PyTorch Metal Performance Shaders (`mps`).
- Precompute target embeddings once to eliminate redundant transformer passes.
- Maintain reproducibility by fixing random seeds (`seed = 42`) for all clustering, splitting, and bootstrapping operations.
