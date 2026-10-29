"""F-Fidelity (Zheng et al., ICLR 2025), adapted to tabular features.

Why not plain 'delete a feature and watch the prediction'? A row with zeros in
places that never occur in training is out-of-distribution; the prediction may
change because the input is strange, not because the feature mattered.

F-Fidelity fixes this in two parts:
  1. fine-tune a SURROGATE copy of the model with random masking (ratio beta),
     so masked inputs are in-distribution for the surrogate;
  2. at evaluation, remove features STOCHASTICALLY, never more than beta*d:
       FFid+ : remove m+ = min(floor(s*alpha), floor(beta*d)) random features
               from the s TOP-ranked ones.     High = faithful.
       FFid- : remove m- = min(ceil((d-s)*alpha), floor(beta*d)) random
               features from the d-s LOWEST-ranked ones.  Near 0 = faithful.
     Each value = 1[y = f(x)] - P(y = f(x_removed)), P over N random draws.
The paper evaluates images, time series and text; this tabular use is our
adaptation (state that in the paper). Masked value = 0 = training mean.

Per-case variant (our addition, for the case-level acceptance test): the paper's
value is accuracy-based, so for ONE case it is mostly 0 or 1 and says little.
ffid_plus_prob uses the same stochastic removal but measures the drop in the
surrogate's P(true class), which is graded per case. Report both, and say which
one the acceptance test uses.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .common import derive_seed


def _correct_prob(model, X, y):
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return (logits.argmax(1).numpy() == y).astype(float)


def _true_prob(model, X, y):
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()
    return p[np.arange(len(y)), y]


def ffidelity_per_case(surrogate, X, y, attr, beta, alpha, n_samples, s, seed):
    """Return per-case dict of arrays: ffid_plus, ffid_minus (paper, accuracy-based)
    and ffid_plus_prob, ffid_minus_prob (probability-based per-case variant)."""
    n, d = X.shape
    budget = int(math.floor(beta * d))
    m_plus = min(int(math.floor(s * alpha)), budget)
    m_minus = min(int(math.ceil((d - s) * alpha)), budget)
    base = _correct_prob(surrogate, X, y)                       # 1[y = f(x)]
    base_p = _true_prob(surrogate, X, y)                        # P(y | x)
    rng = np.random.default_rng(derive_seed(seed, "ffid"))
    order = np.argsort(-np.abs(attr), axis=1, kind="stable")    # per-case ranking
    top, bottom = order[:, :s], order[:, s:]

    def removed(pool, m):
        Xr = np.repeat(X, n_samples, axis=0).copy()            # (n*N, d)
        for i in range(n):
            for j in range(n_samples):
                cols = rng.choice(pool[i], size=m, replace=False)
                Xr[i * n_samples + j, cols] = 0.0
        yr = np.repeat(y, n_samples)
        acc = _correct_prob(surrogate, Xr, yr).reshape(n, n_samples).mean(1)
        prob = _true_prob(surrogate, Xr, yr).reshape(n, n_samples).mean(1)
        return acc, prob

    acc_p, prob_p = removed(top, m_plus)
    acc_m, prob_m = removed(bottom, m_minus)
    out = {"ffid_plus": base - acc_p, "ffid_minus": base - acc_m,
           "ffid_plus_prob": base_p - prob_p, "ffid_minus_prob": base_p - prob_m}
    return out, dict(m_plus=m_plus, m_minus=m_minus, budget=budget)


def naive_deletion_per_case(model, X, y, attr, s):
    """Baseline (OOD-prone): ORIGINAL model, set ALL top-s features to 0 at once.
    Returns drop in P(true class)."""
    order = np.argsort(-np.abs(attr), axis=1, kind="stable")[:, :s]
    Xr = X.copy()
    np.put_along_axis(Xr, order, 0.0, axis=1)
    with torch.no_grad():
        p0 = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), 1).numpy()[np.arange(len(y)), y]
        p1 = torch.softmax(model(torch.tensor(Xr, dtype=torch.float32)), 1).numpy()[np.arange(len(y)), y]
    return p0 - p1
