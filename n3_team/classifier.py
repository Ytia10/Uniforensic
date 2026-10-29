"""
classifier.py
Explainable Machine Learning Classifier with Platt Sigmoid Probability Calibration.
Trained on spec-derived SQLite forensic features.
Calculates calibrated confidence scores, Expected Calibration Error (ECE), and Brier Score.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple, Optional


class CalibratedSQLiteClassifier:
    """
    Lightweight, explainable linear classifier with Platt Sigmoid Calibration.
    Converts spec-derived forensic features into well-calibrated posterior probabilities:
    P(real_record | features) where 0.8 means roughly 80% empirical precision.
    """
    FEATURE_NAMES = [
        "f1_varint_valid",
        "f2_serial_legal",
        "f3_schema_match",
        "f4_length_fit",
        "f5_text_validity",
        "f6_value_sanity",
        "f7_entropy_score",
        "f8_freeblock_alignment"
    ]

    def __init__(self, l2_reg: float = 0.01):
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = 0.0
        # Platt calibration parameters: P(y=1 | s) = 1 / (1 + exp(A*s + B))
        self.platt_a = -1.0
        self.platt_b = 0.0
        self.is_fitted = False

    def _features_to_matrix(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        matrix = []
        for fd in feature_dicts:
            row = [float(fd.get(fname, 0.0)) for fname in self.FEATURE_NAMES]
            matrix.append(row)
        return np.array(matrix, dtype=np.float64)

    def fit(self, X_dicts: List[Dict[str, float]], y: List[int], val_split: float = 0.25):
        """
        Fits weights using regularized logistic loss, then fits Platt sigmoid on validation split.
        """
        X = self._features_to_matrix(X_dicts)
        y = np.array(y, dtype=np.float64)
        n_samples, n_features = X.shape

        if n_samples < 4:
            # Fallback for tiny sample
            self.weights = np.ones(n_features) / n_features
            self.bias = 0.0
            self.is_fitted = True
            return

        # Train / validation split for Platt calibration
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1.0 - val_split))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 1. Train base logistic regression
        def loss_func(params):
            w = params[:n_features]
            b = params[n_features]
            logits = np.dot(X_train, w) + b
            # Numerical stability: log(1 + exp(-z))
            loss = np.sum(np.log1p(np.exp(-logits)) + (1.0 - y_train) * logits)
            reg = 0.5 * self.l2_reg * np.sum(w ** 2)
            return loss / len(y_train) + reg

        initial_params = np.zeros(n_features + 1)
        res = minimize(loss_func, initial_params, method="L-BFGS-B")
        self.weights = res.x[:n_features]
        self.bias = res.x[n_features]

        # 2. Fit Platt calibration on validation set
        val_logits = np.dot(X_val, self.weights) + self.bias

        # Platt scaling optimization: minimize binary cross-entropy over (A, B)
        # Target probability mapping with Laplace smoothing
        n_pos = np.sum(y_val == 1)
        n_neg = np.sum(y_val == 0)
        t_val = np.where(y_val == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        def platt_loss(params):
            A, B = params[0], params[1]
            p = 1.0 / (1.0 + np.exp(A * val_logits + B))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return -np.mean(t_val * np.log(p) + (1.0 - t_val) * np.log(1.0 - p))

        platt_res = minimize(platt_loss, [-1.0, 0.0], method="L-BFGS-B")
        self.platt_a = float(platt_res.x[0])
        self.platt_b = float(platt_res.x[1])
        self.is_fitted = True

    def predict_proba(self, X_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Returns calibrated probabilities P(y=1 | x) in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X = self._features_to_matrix(X_dicts)
        raw_logits = np.dot(X, self.weights) + self.bias
        # Platt calibrated probability
        calibrated_probs = 1.0 / (1.0 + np.exp(self.platt_a * raw_logits + self.platt_b))
        return np.clip(calibrated_probs, 0.0, 1.0)

    def evaluate_calibration(self, probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10) -> Dict[str, Any]:
        """
        Computes reliability diagram data, Expected Calibration Error (ECE), and Brier Score.
        """
        probs = np.array(probs)
        y_true = np.array(y_true)

        brier_score = float(np.mean((probs - y_true) ** 2))

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        diagram_data = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs >= lower) & (probs < upper)
            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = float(np.mean(y_true[in_bin]))
                bin_conf = float(np.mean(probs[in_bin]))
                ece += (bin_size / len(probs)) * abs(bin_acc - bin_conf)
                diagram_data.append({
                    "bin_range": (float(lower), float(upper)),
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                    "count": int(bin_size)
                })
            else:
                diagram_data.append({
                    "bin_range": (float(lower), float(upper)),
                    "accuracy": 0.0,
                    "confidence": (lower + upper) / 2.0,
                    "count": 0
                })

        return {
            "ece": float(ece),
            "brier_score": float(brier_score),
            "reliability_diagram": diagram_data
        }
