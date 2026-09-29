"""Scoring for held-out-target evaluation.

The distinction this module exists to enforce: a correlation computed over all
held-out pairs at once and a correlation computed inside each target and then
averaged are different measurements, and only the second one answers "can this
model rank peptides for a target it has never seen".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def pooled_spearman(targets, y_true, y_pred) -> float:
    """Rank every held-out pair against every other, ignoring which target it came from.

    This is what the published peptide-affinity benchmarks report. It is kept
    here because we want to show the gap between it and the per-target number,
    not because it answers a question anyone has.
    """
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        return None
    return float(spearmanr(y, p).statistic)


def per_target_spearman(targets, y_true, y_pred, minimum: int = 10) -> dict:
    """Rank peptides inside each target, then average over targets.

    A target whose measured affinities are all equal orders nothing and is
    dropped; a prediction that is constant within a target orders nothing and
    scores 0, because that is a real failure rather than an undefined one.
    """
    frame = pd.DataFrame({"target": targets, "y": y_true, "p": y_pred})
    scores = {}
    for target, group in frame.groupby("target"):
        if len(group) < minimum or np.ptp(group["y"]) == 0:
            continue
        scores[target] = 0.0 if np.ptp(group["p"]) == 0 else float(
            spearmanr(group["y"], group["p"]).statistic
        )
    usable = [v for v in scores.values() if np.isfinite(v)]
    return {
        "macro_spearman": float(np.mean(usable)) if usable else None,
        "targets_scored": len(usable),
        "per_target": scores,
    }


def per_target_skill(targets, y_true, y_pred, minimum: int = 10) -> dict:
    """Absolute error against each target's own median affinity.

    This is the calibration half. Ranking is invariant to any monotone rescaling,
    so a model can order peptides perfectly and still place every prediction in
    the wrong part of the affinity scale. The reference is the target's own
    median because that is the best constant an oracle could pick knowing only
    that the target exists, so skill is 0 for a model that knows nothing about
    the scale and negative for one that is confidently wrong about it.

    Note what this reference concedes: it is computed from the held-out target's
    own labels, which a deployed model would not have. It is therefore a
    generous reference, and a model that cannot beat it has no scale knowledge
    at all.
    """
    frame = pd.DataFrame({"target": targets, "y": y_true, "p": y_pred})
    scores = {}
    for target, group in frame.groupby("target"):
        if len(group) < minimum:
            continue
        reference = float(np.mean(np.abs(group["y"] - group["y"].median())))
        if reference == 0:
            continue
        model = float(np.mean(np.abs(group["y"] - group["p"])))
        scores[target] = float(1 - model / reference)
    usable = list(scores.values())
    return {
        "macro_skill": float(np.mean(usable)) if usable else None,
        "targets_scored": len(usable),
        "per_target": scores,
    }


def per_target_enrichment(targets, y_true, y_pred, top: float = 0.2, minimum: int = 10) -> dict:
    """How many of the tightest binders a model puts in its own shortlist.

    A screen tests a fixed number of wells, so the decision-relevant question is
    not the whole ordering but the head of it. This reports the fraction of the
    true top `top` captured in the predicted top `top`, divided by `top` itself,
    so 1.0 is what random selection gives and 2.0 is twice as many hits per well.

    The shortlist size is rounded up, so for 11 peptides it is 3 rather than
    2.2. The random expectation is therefore k/n rather than `top`, and that is
    what the ratio divides by; dividing by `top` would score a coin flip at 1.36
    on such a target.

    A prediction that is constant within a target has no shortlist of its own,
    so it scores exactly 1.0 rather than whatever the arbitrary tie order gives.
    """
    frame = pd.DataFrame({"target": targets, "y": y_true, "p": y_pred})
    scores = {}
    for target, group in frame.groupby("target"):
        if len(group) < minimum or np.ptp(group["y"]) == 0:
            continue
        n = len(group)
        k = int(np.ceil(top * n))
        if np.ptp(group["p"]) == 0:
            scores[target] = 1.0
            continue
        best = set(group["y"].nlargest(k).index)
        chosen = set(group["p"].nlargest(k).index)
        scores[target] = (len(best & chosen) / k) / (k / n)
    usable = list(scores.values())
    return {
        "macro_enrichment": float(np.mean(usable)) if usable else None,
        "targets_scored": len(usable),
        "per_target": scores,
    }


def report(targets, y_true, y_pred, minimum: int = 10) -> dict:
    """Both axes at once, so neither can be quoted without the other."""
    rank = per_target_spearman(targets, y_true, y_pred, minimum)
    skill = per_target_skill(targets, y_true, y_pred, minimum)
    lift = per_target_enrichment(targets, y_true, y_pred, minimum=minimum)
    return {
        "pooled_spearman": pooled_spearman(targets, y_true, y_pred),
        "macro_spearman": rank["macro_spearman"],
        "macro_enrichment": lift["macro_enrichment"],
        "macro_skill": skill["macro_skill"],
        "targets_scored": rank["targets_scored"],
    }
