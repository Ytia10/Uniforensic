"""Explanation scores.

Ground-truth metrics (need the planted answer key G):
    precision@k, recall@k  - top-k features by |attribution| vs the planted set
    sign_agreement_gt      - share of planted features attributed POSITIVELY
                             (planted evidence should push towards the class)
    mass_on_gt             - share of total |attribution| that lands on G

Explainer-vs-explainer agreement (Krishna et al., TMLR 2024; also in OpenXAI):
    FA  feature agreement, RA rank agreement, SA sign agreement,
    SRA signed rank agreement (all @k), RC rank correlation, PRA pairwise rank agreement
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr


def topk_idx(a: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest |a|, ties broken by feature order (deterministic)."""
    return np.argsort(-np.abs(a), kind="stable")[:k]


# ---------------------------------------------------------------- vs answer key
def precision_at_k(a, G: set, k: int) -> float:
    return len(set(topk_idx(a, k)) & G) / k


def recall_at_k(a, G: set, k: int) -> float:
    return len(set(topk_idx(a, k)) & G) / len(G) if G else np.nan


def sign_agreement_gt(a, G: set) -> float:
    return float(np.mean([a[i] > 0 for i in G])) if G else np.nan


def mass_on_gt(a, G: set) -> float:
    tot = np.abs(a).sum()
    return float(np.abs(a[list(G)]).sum() / tot) if tot > 0 and G else np.nan


# ------------------------------------------------------- explainer vs explainer
def feature_agreement(a, b, k):
    return len(set(topk_idx(a, k)) & set(topk_idx(b, k))) / k


def rank_agreement(a, b, k):
    return float(np.mean(topk_idx(a, k) == topk_idx(b, k)))


def sign_agreement(a, b, k):
    common = set(topk_idx(a, k)) & set(topk_idx(b, k))
    return sum(np.sign(a[i]) == np.sign(b[i]) for i in common) / k


def signed_rank_agreement(a, b, k):
    ta, tb = topk_idx(a, k), topk_idx(b, k)
    return sum((ta[j] == tb[j]) and (np.sign(a[ta[j]]) == np.sign(b[tb[j]])) for j in range(k)) / k


def rank_correlation(a, b):
    r = spearmanr(np.abs(a), np.abs(b)).statistic
    return float(0.0 if np.isnan(r) else r)


def pairwise_rank_agreement(a, b):
    aa, bb = np.abs(a), np.abs(b)
    pairs = list(itertools.combinations(range(len(a)), 2))
    return float(np.mean([(aa[i] > aa[j]) == (bb[i] > bb[j]) for i, j in pairs]))


PAIRWISE = {
    "FA": lambda a, b, k: feature_agreement(a, b, k),
    "RA": lambda a, b, k: rank_agreement(a, b, k),
    "SA": lambda a, b, k: sign_agreement(a, b, k),
    "SRA": lambda a, b, k: signed_rank_agreement(a, b, k),
    "RC": lambda a, b, k: rank_correlation(a, b),
    "PRA": lambda a, b, k: pairwise_rank_agreement(a, b),
}


# ------------------------------------------------------------------- statistics
def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0):
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------- self test
def self_test() -> None:
    """Hand-made cases with known answers. Raises AssertionError if code is wrong."""
    a = np.array([0.9, -0.5, 0.1, 0.0, 0.3])      # ranking by |a|: 0, 1, 4, 2, 3
    G = {0, 4}
    assert list(topk_idx(a, 3)) == [0, 1, 4]
    assert precision_at_k(a, G, 2) == 0.5          # top-2 = {0,1}; 0 in G
    assert recall_at_k(a, G, 3) == 1.0             # top-3 contains 0 and 4
    assert sign_agreement_gt(a, G) == 1.0
    assert abs(mass_on_gt(a, G) - 1.2 / 1.8) < 1e-12
    b = np.array([0.8, 0.6, 0.0, 0.0, 0.2])        # same ranking, sign of feature 1 flipped
    assert feature_agreement(a, b, 3) == 1.0
    assert rank_agreement(a, b, 3) == 1.0
    assert abs(sign_agreement(a, b, 3) - 2 / 3) < 1e-12
    assert abs(signed_rank_agreement(a, b, 3) - 2 / 3) < 1e-12
    assert feature_agreement(a, -a, 5) == 1.0 and sign_agreement(a, -a, 5) == 0.2   # only the 0 matches
    assert pairwise_rank_agreement(a, a) == 1.0
    assert abs(rank_correlation(a, a) - 1.0) < 1e-12
    m, lo, hi = bootstrap_ci([1.0] * 10)
    assert m == lo == hi == 1.0
