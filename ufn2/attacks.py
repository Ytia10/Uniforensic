"""Membership-inference attacks on explanations.

On SHARED SUMMARIES (Part A):
  tracing_score  - inner-product test of Dwork, Smith, Steinke, Ullman & Vadhan
                   ("Robust traceability from trace amounts", FOCS 2015):
                   score = < x_target - w , released_mean - w >, where w is the
                   average of the attacker's own reference cases. A member pulls
                   the released mean towards itself, so its score is higher.
  distinguishability game - Pyrgelis, Troncoso & De Cristofaro (NDSS 2018):
                   the attacker trains a classifier (logistic regression, random
                   forest) on aggregates it builds itself WITH and WITHOUT the
                   target, then guesses on real aggregates. Success = AUC.

On PER-CASE EXPLANATIONS (Part B):
  statistics of the attribution vector: variance (Shokri, Strobel & Zick, AIES 2021),
  L1 and L2 norms (Huang, Pawelczyk & Lakkaraju, 2024); used as a global
  threshold test and as a shadow-model-calibrated likelihood-ratio-style test.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ metrics
def auc(scores_in, scores_out) -> float:
    y = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    return float(roc_auc_score(y, np.r_[scores_in, scores_out]))


def auc_ci(scores_in, scores_out, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval of the AUC (members and non-members resampled)."""
    si, so = np.asarray(scores_in), np.asarray(scores_out)
    rng = np.random.default_rng(seed)
    vals = [auc(rng.choice(si, len(si)), rng.choice(so, len(so))) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tpr_at_fpr(scores_in, scores_out, fpr: float) -> float:
    """True-positive rate when at most `fpr` of non-members are wrongly flagged."""
    y = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    f, t, _ = roc_curve(y, np.r_[scores_in, scores_out])
    return float(t[f <= fpr].max()) if (f <= fpr).any() else 0.0


# ------------------------------------------------------- tracing (Dwork et al.)
def tracing_score(x_target: np.ndarray, released_mean: np.ndarray, w: np.ndarray) -> float:
    return float((x_target - w) @ (released_mean - w))


# ------------------------------------------------ distinguishability (Pyrgelis)
def make_classifiers(seed: int, rf_trees: int):
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "rf": RandomForestClassifier(n_estimators=rf_trees, random_state=seed, n_jobs=1)}


def game_aggregates(target: np.ndarray, pool: np.ndarray, m: int, n_games: int, C: float,
                    sigma: float, rng):
    """n_games aggregates of m cases: with b=1 the target is in the group, with b=0
    it is replaced by another random case (Pyrgelis et al.'s game). Returns (X, b)."""
    X, b = np.zeros((n_games, pool.shape[1])), rng.integers(0, 2, n_games)
    for g in range(n_games):
        idx = rng.choice(len(pool), size=m, replace=False)
        group = pool[idx].copy()
        if b[g] == 1:
            group[0] = target                    # target takes one seat in the group
        S, n = group.sum(0), float(m)
        if sigma > 0:
            S = S + rng.normal(0, sigma * C, S.shape)
            n = n + rng.normal(0, sigma)
        X[g] = S / max(n, 1.0)
    return X, b


# ---------------------------------------------------- per-case explanation stats
def explanation_stats(attr: np.ndarray) -> dict[str, np.ndarray]:
    return {"variance": attr.var(axis=1), "l1": np.abs(attr).sum(axis=1),
            "l2": np.linalg.norm(attr, axis=1)}


def self_test(seed: int = 0) -> None:
    """Attacks must work where leakage is certain and fail where it is impossible."""
    rng = np.random.default_rng(seed)
    pop = np.abs(rng.normal(size=(5000, 27)))
    w = pop[:2000].mean(0)
    group, outs = pop[2000:2005], pop[3000:3005]               # 5 members, 5 non-members
    mean = group.mean(0)
    s_in = [tracing_score(x, mean, w) for x in group]
    s_out = [tracing_score(x, mean, w) for x in outs]
    assert auc(s_in, s_out) > 0.9                               # tiny group, no noise: leaks
    big = pop[2000:2500]                                        # 500 members
    noisy = big.mean(0) + rng.normal(0, 50, 27)                 # huge noise: no signal
    s_in = [tracing_score(x, noisy, w) for x in big]
    s_out = [tracing_score(x, noisy, w) for x in pop[3000:3500]]
    assert abs(auc(s_in, s_out) - 0.5) < 0.1
    assert tpr_at_fpr([2, 3], [0, 1], 0.0) == 1.0
