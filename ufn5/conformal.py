"""Split-conformal prediction sets (LAC score; Sadinle, Lei & Wasserman, JASA 2019).

score(x, y) = 1 - P(y | x).  With n calibration cases, q = the ceil((n+1)(1-alpha))-th
smallest score; the set for a new case is every class with P(c | x) >= 1 - q.
If calibration and test cases are exchangeable, the set contains the true class
with probability >= 1 - alpha (Vovk et al., 2005; Angelopoulos & Bates, 2023).

N5 builds TWO sets per case from the same model:
  full set - from the input with the recovered records
  live set - from the input with every recovered record removed
Foreign residue only ever enters through recovered records, so it cannot change the
distribution of live inputs: the live set keeps its guarantee whatever the residue
rate, while the full set's guarantee needs the residue rate at test time to match
calibration. Reporting both shows the examiner how much the recovered records
narrow the conclusion, with a guarantee that does not depend on them.

MAPIE (SplitConformalClassifier, conformity_score='lac', prefit=True) is used as an
independent cross-check of this implementation (n5_08).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def lac_quantile(P_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    s = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return 1.0 if k > n else float(np.sort(s)[k - 1])


def lac_sets(P: np.ndarray, q: float) -> np.ndarray:
    """Boolean (n_cases, n_classes): class c is in the set if 1 - P_c <= q."""
    return (1.0 - P) <= q


def coverage(sets: np.ndarray, y: np.ndarray) -> float:
    return float(sets[np.arange(len(y)), y].mean())


class FixedProba(ClassifierMixin, BaseEstimator):
    """A 'prefit' scikit-learn classifier that returns pre-computed probabilities for
    row i of an index matrix - lets MAPIE score exactly the same numbers."""

    def __init__(self, proba: np.ndarray | None = None):
        self.proba = proba

    def fit(self, X, y=None):
        self.classes_ = np.arange(self.proba.shape[1])
        return self

    def predict_proba(self, X):
        return self.proba[np.asarray(X)[:, 0].astype(int)]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def __sklearn_is_fitted__(self):
        return True


def mapie_sets(P_all: np.ndarray, cal_idx: np.ndarray, y_cal: np.ndarray, test_idx: np.ndarray,
               alpha: float) -> np.ndarray:
    from mapie.classification import SplitConformalClassifier
    est = FixedProba(P_all).fit(None)
    m = SplitConformalClassifier(estimator=est, confidence_level=1 - alpha,
                                 conformity_score="lac", prefit=True)
    m.conformalize(cal_idx.reshape(-1, 1), y_cal)
    _, sets = m.predict_set(test_idx.reshape(-1, 1))
    return sets[:, :, 0]
