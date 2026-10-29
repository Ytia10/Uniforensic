"""Non-IID split of cases across agencies with a Dirichlet distribution.

Same method as Flower Datasets' DirichletPartitioner (partition by label;
Hsu et al., 2019): for every class, draw the share each agency gets from
Dirichlet(alpha, ..., alpha). Small alpha -> each agency sees mostly a few
classes; large alpha -> every agency gets about the same mix. If any agency
ends up with fewer than `min_size` cases, the draw is repeated (as Flower does).
Implemented here (about 20 lines) to avoid Flower Datasets' heavy
HuggingFace dependency.
"""
from __future__ import annotations

import numpy as np


def dirichlet_partition(labels: np.ndarray, n_agencies: int, alpha: float, seed: int,
                        min_size: int = 30, max_tries: int = 100) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_agencies)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            share = rng.dirichlet(np.full(n_agencies, alpha))
            cuts = (np.cumsum(share) * len(idx)).astype(int)[:-1]
            for a, chunk in enumerate(np.split(idx, cuts)):
                parts[a].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError(f"No split with >= {min_size} cases per agency after {max_tries} tries "
                       f"(alpha={alpha}). Lower min_cases_per_agency in config_n4.yaml.")


def class_counts(parts: list[np.ndarray], labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Matrix (agencies x classes) of case counts."""
    return np.array([[int((labels[p] == c).sum()) for c in range(n_classes)] for p in parts])
