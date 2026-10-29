"""Recalibrate the teammate's N3 confidence on N1-derived databases.

N3's Platt scaling was fit on N3's own 4 training databases (plus synthetic
negatives), and N3 reports its ECE on that same training data. Whether p still
means "probability that this record really existed" on OTHER databases is an
empirical question, so N5 measures it on held-out N1 databases and, if needed,
learns a monotone correction (isotonic regression; Zadrozny & Elkan, KDD 2002)
per table on val_fit. Monotone = the teammate's ranking of records is kept.

Calibrated confidence is rounded to whole percent: CASE/UCO's core:confidence is
an integer 0-100, so the value travels through a CASE export without loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

GENUINE = ("correct", "foreign")    # the record really existed (who owns it is another question)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Same binning as the teammate's CalibratedSQLiteClassifier.evaluate_calibration
    (bins [lo, hi), so p = 1.0 falls outside the last bin there; here the last bin is
    closed so that every record counts)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_isotonic(cal: dict, p: np.ndarray) -> np.ndarray:
    """Equivalent to IsotonicRegression.predict (linear interpolation, clipped)."""
    return np.interp(np.asarray(p, float), cal["x"], cal["y"])


def to_percent(p: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(p, float) * 100.0) / 100.0
