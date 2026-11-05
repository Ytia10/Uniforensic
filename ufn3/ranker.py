"""The learned part of N3: rank recovered records and turn scores into probabilities.

Model: scikit-learn HistGradientBoostingClassifier on the spec-derived features of
carver.MODEL_FEATURES (fast, deterministic with a fixed random_state, handles mixed
scales without preprocessing).
Calibration: a SEPARATE split (built from N1's validation cases) fits an isotonic
map from score to probability, with the trained model frozen (scikit-learn
FrozenEstimator + CalibratedClassifierCV). scikit-learn's guide advises isotonic
only with more than about 1,000 calibration samples; the calibration split has
over 10,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from .carver import MODEL_FEATURES


_TEXT_COLS = {c: str for c in ("artifact_id", "role", "pattern_id", "values", "table", "case_id")}


def read_records(path) -> pd.DataFrame:
    """Load a records file written by step 4 (text columns kept as text)."""
    return pd.read_csv(path, dtype=_TEXT_COLS)


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Records that could be reported: live duplicates are excluded by construction."""
    return df[df["live_dup"] == 0].reset_index(drop=True)


def matrix(df: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> np.ndarray:
    return df[features].to_numpy(dtype=float)


def fit_ranker(df: pd.DataFrame, rcfg: dict, seed: int,
               features: list[str] = MODEL_FEATURES) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(max_iter=rcfg["max_iter"], learning_rate=rcfg["learning_rate"],
                                       max_leaf_nodes=rcfg["max_leaf_nodes"],
                                       min_samples_leaf=rcfg["min_samples_leaf"],
                                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                       random_state=seed)
    return m.fit(matrix(df, features), df["y"].to_numpy())


def calibrate(model, df_cal: pd.DataFrame, method: str,
              features: list[str] = MODEL_FEATURES) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    return cal.fit(matrix(df_cal, features), df_cal["y"].to_numpy())


def threshold_for_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Lowest threshold t such that reporting every record with p >= t reaches the target
    precision on the calibration split (ties in p are reported together); 1.0 if none."""
    y, p = np.asarray(y), np.asarray(p)
    best = 1.0
    for t in np.unique(p)[::-1]:
        rep = p >= t
        if y[rep].mean() >= target:
            best = float(t)
    return best
