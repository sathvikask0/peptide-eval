"""Models fitted on frozen embeddings, and the controls they must beat.

Each model is defined by which features it is allowed to see, because that is
what the experiment is about: a model that sees only the target cannot rank
peptides, and one that sees only the peptide cannot condition on the target.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Regularisation is chosen per model by internal cross-validation over a wide
# grid rather than fixed, so a model is not handicapped by a constant that
# happened to suit a different feature width.
ALPHAS = np.logspace(-1, 6, 15)


def features(kind: str, target_vectors: np.ndarray, peptide_vectors: np.ndarray) -> np.ndarray:
    """Build the design matrix a given model is permitted to see."""
    if kind == "target_only":
        return target_vectors
    if kind == "peptide_only":
        return peptide_vectors
    if kind == "concat":
        return np.hstack([target_vectors, peptide_vectors])
    if kind == "interaction":
        # The element-wise product is the cheapest way to let the model express
        # "this peptide suits this target" rather than "this peptide is good".
        # Concatenation alone cannot: a linear model on concatenated features is
        # a sum of a target term and a peptide term, so its ranking of peptides
        # is identical for every target.
        return np.hstack([target_vectors, peptide_vectors, target_vectors * peptide_vectors])
    raise ValueError(f"unknown model {kind}")


class RidgeModel:
    """Ridge on standardised features.

    Ridge rather than a network on purpose. The question is whether these
    representations carry target-conditioned binding information at all, and a
    linear probe answers that without the confound of how long something trained
    or which architecture it had. A network that beat it would be a later
    result, not this one.
    """

    def __init__(self, kind: str):
        self.kind = kind
        self.scaler = StandardScaler()
        self.ridge = RidgeCV(alphas=ALPHAS)

    def fit(self, target_vectors, peptide_vectors, y):
        x = self.scaler.fit_transform(features(self.kind, target_vectors, peptide_vectors))
        self.ridge.fit(x, y)
        return self

    def predict(self, target_vectors, peptide_vectors):
        x = self.scaler.transform(features(self.kind, target_vectors, peptide_vectors))
        return self.ridge.predict(x)
