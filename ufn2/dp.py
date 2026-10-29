"""Gaussian noise calibrated EXACTLY for (epsilon, delta)-differential privacy.

The textbook formula sigma = Delta * sqrt(2 ln(1.25/delta)) / epsilon is only valid
for epsilon < 1, and our sweep goes up to epsilon = 8. We therefore use the
analytic Gaussian mechanism (Balle & Wang, ICML 2018, Theorem 8):

  adding N(0, sigma^2) noise to a function with L2 sensitivity Delta is
  (epsilon, delta)-DP  if and only if
      Phi( Delta/(2 sigma) - epsilon sigma/Delta )
    - e^epsilon * Phi( -Delta/(2 sigma) - epsilon sigma/Delta )   <=  delta

`sigma_for` finds the smallest such sigma by bisection.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def delta_of(eps: float, sigma: float, sens: float) -> float:
    """The smallest delta for which N(0, sigma^2) gives (eps, delta)-DP."""
    a, b = sens / (2 * sigma), eps * sigma / sens
    # e^eps * Phi(-a-b) computed in log space to avoid overflow for large eps
    return float(norm.cdf(a - b) - math.exp(eps + norm.logcdf(-a - b)))


def sigma_for(eps: float, delta: float, sens: float) -> float:
    """Smallest sigma giving (eps, delta)-DP. eps = inf -> 0 (no noise)."""
    if math.isinf(eps):
        return 0.0
    lo, hi = 1e-6 * sens, 1e6 * sens
    for _ in range(200):                       # delta_of decreases as sigma grows
        mid = math.sqrt(lo * hi)
        if delta_of(eps, mid, sens) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def self_test(seed: int = 0) -> None:
    """Checks with known answers. Raises AssertionError if the code is wrong."""
    # 1. the condition holds at the returned sigma, and fails just below it
    for eps in (0.5, 1, 2, 4, 8):
        s = sigma_for(eps, 1e-5, 1.0)
        assert delta_of(eps, s, 1.0) <= 1e-5 + 1e-12
        assert delta_of(eps, s * 0.99, 1.0) > 1e-5
    # 2. for eps < 1 the analytic sigma is never larger than the classical one
    for eps in (0.1, 0.5, 0.9):
        classical = math.sqrt(2 * math.log(1.25 / 1e-5)) / eps
        assert sigma_for(eps, 1e-5, 1.0) <= classical
    # 3. noise scales linearly with sensitivity
    assert abs(sigma_for(1, 1e-5, 2.0) / sigma_for(1, 1e-5, 1.0) - 2) < 1e-6
    # 4. Monte-Carlo check of the DP definition itself. For P = N(0, s^2) and
    #    Q = N(Delta, s^2) the privacy loss is L(x) = log p(x)/q(x)
    #    = (Delta^2 - 2 x Delta) / (2 s^2), and the tight delta is
    #    E_{x~P}[ max(0, 1 - e^(eps - L(x))) ].
    rng = np.random.default_rng(seed)
    eps, sens = 1.0, 1.0
    s = sigma_for(eps, 1e-3, sens)
    x = rng.normal(0.0, s, 4_000_000)
    loss = (sens ** 2 - 2 * x * sens) / (2 * s ** 2)
    mc = float(np.mean(np.clip(1 - np.exp(eps - loss), 0, None)))
    assert abs(mc - 1e-3) < 1e-4, mc
