"""What an agency shares about its explanations, and how it can be protected.

Each case i gets an explanation vector a_i (Integrated Gradients, 27 features,
explaining the model's predicted class). An agency shares, PER PREDICTED CLASS c:
    S_c = sum over its cases of clip(|a_i|)      (27 numbers)
    n_c = how many of its cases fall in class c   (1 number)
The receiver reads the mean S_c / n_c: "on our fraud cases, the model relied on
these features this much". No case-level data is shared.

clip(v) rescales v so that ||v||_2 <= C. This bounds what ONE case can change,
which is what differential privacy needs (sensitivity).

Release schemes:
  none       - the server sees every agency's (S_c, n_c) as is
  dp         - each agency adds Gaussian noise before sending
  secagg     - secure aggregation: the server only sees the SUM over the 8 agencies
               (Flower's SecAgg+ computes this sum exactly, up to its quantisation
               error of at most ~3e-5 per value with 8 agencies; we model the sum)
  secagg_dp  - secure aggregation + noise: each agency adds N(0, sigma^2 / K), so the
               sum carries N(0, sigma^2) in total ("distributed DP")

Sensitivity: one case changes one class block by at most C in S_c and by 1 in n_c.
We release (S_c / C, n_c), so the L2 sensitivity is sqrt(1^2 + 1^2) = sqrt(2).
"""
from __future__ import annotations

import math

import numpy as np

from .dp import sigma_for

SENSITIVITY = math.sqrt(2.0)


def clip_rows(v: np.ndarray, C: float) -> np.ndarray:
    """|attributions| with each row rescaled to L2 norm <= C."""
    v = np.abs(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, C / np.maximum(norms, 1e-12))


def block_sums(vecs: np.ndarray, classes: np.ndarray, n_classes: int):
    """Per-class (sum of vectors, count) for one group of cases."""
    S = np.zeros((n_classes, vecs.shape[1]))
    n = np.zeros(n_classes)
    for c in range(n_classes):
        m = classes == c
        S[c], n[c] = vecs[m].sum(0), m.sum()
    return S, n


def noisy_mean(S: np.ndarray, n: np.ndarray, C: float, sigma: float, rng) -> np.ndarray:
    """Add N(0, sigma^2) to (S/C, n) and return the mean the receiver computes.
    sigma is in units of the scaled release; sigma = 0 means no noise."""
    if sigma > 0:
        S = S + rng.normal(0.0, sigma * C, S.shape)
        n = n + rng.normal(0.0, sigma, n.shape)
    return S / np.maximum(n, 1.0)[..., None]


def release(scheme: str, agency_blocks: list[tuple[np.ndarray, np.ndarray]], C: float,
            eps: float, delta: float, rng):
    """Return what the server sees.
    none/dp        -> list of per-agency mean matrices (classes x features)
    secagg/_dp     -> one global mean matrix
    """
    sigma = sigma_for(eps, delta, SENSITIVITY) if scheme in ("dp", "secagg_dp") else 0.0
    if scheme in ("none", "dp"):
        return [noisy_mean(S, n, C, sigma, rng) for S, n in agency_blocks]
    K = len(agency_blocks)
    S_tot = sum(S for S, _ in agency_blocks)
    n_tot = sum(n for _, n in agency_blocks)
    if scheme == "secagg_dp":
        # each agency adds N(0, sigma^2/K); the K shares sum to N(0, sigma^2)
        S_tot = S_tot + sum(rng.normal(0, sigma * C / math.sqrt(K), S_tot.shape) for _ in range(K))
        n_tot = n_tot + sum(rng.normal(0, sigma / math.sqrt(K), n_tot.shape) for _ in range(K))
    return S_tot / np.maximum(n_tot, 1.0)[..., None]


def evidence_recall_at_3(mean_vec: np.ndarray, decisive_idx: list[int]) -> float:
    """Do the 3 largest values of a class summary sit on that class's 3 planted
    decisive features (N1 answer key)? 1.0 = all three, 0.33 = one."""
    top = np.argsort(-mean_vec, kind="stable")[:3]
    return len(set(top) & set(decisive_idx)) / 3.0
