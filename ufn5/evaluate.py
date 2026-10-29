"""Scoring helpers shared by N5 steps 8-11."""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients
from sklearn.metrics import f1_score

from ufn1.metrics import recall_at_k
from ufn1.models import ProbWrapper, TriageMLP
from ufn1.schema import CLASSES, DECISIVE_FEATURES, FEATURES

BENIGN = CLASSES.index("benign")
DEC_IDX = [FEATURES.index(f) for f in DECISIVE_FEATURES]   # = 0..8 (decisive features come first)
CHUNK = 250


def load_mlp(cfg1: dict, path, n_in: int) -> TriageMLP:
    m = TriageMLP(n_in, len(CLASSES), cfg1["mlp"]["hidden"], cfg1["mlp"]["dropout"])
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def triage_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    benign = y == BENIGN
    return {"n": int(len(y)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(CLASSES)))),
            "accuracy": float((pred == y).mean()),
            # an innocent person's device triaged as an offence case
            "false_incrimination": float((pred[benign] != BENIGN).mean()) if benign.any() else np.nan,
            "offence_recall": float((pred[~benign] == y[~benign]).mean()) if (~benign).any() else np.nan}


def integrated_gradients(model, X: np.ndarray, targets: np.ndarray, baseline: np.ndarray,
                         n_steps: int) -> np.ndarray:
    """Captum IG on P(target | case), exactly as N1's run_explainer, but with an explicit
    baseline row (N1: all zeros = training mean). Variant C uses 'nothing recovered'
    as the baseline of its recovered channel."""
    ig = IntegratedGradients(ProbWrapper(model).eval())
    out = []
    for i in range(0, len(X), CHUNK):          # chunks: same result, linear time (see ufn2.common)
        Xt = torch.tensor(X[i:i + CHUNK], dtype=torch.float32)
        base = torch.tensor(np.broadcast_to(baseline, Xt.shape).copy(), dtype=torch.float32)
        t = torch.tensor(targets[i:i + CHUNK], dtype=torch.long)
        out.append(ig.attribute(Xt, baselines=base, target=t, n_steps=n_steps).detach().numpy())
    return np.vstack(out).astype(np.float64)


def to_base(attr: np.ndarray) -> np.ndarray:
    """36 C attributions -> 27 base features (live + recovered channel of the same feature),
    so every variant is scored against the same N1 answer key."""
    if attr.shape[1] == len(FEATURES):
        return np.abs(attr)
    a = np.abs(attr[:, :len(FEATURES)]).copy()
    a[:, DEC_IDX] += np.abs(attr[:, len(FEATURES):])
    return a


def recovered_share(attr: np.ndarray) -> np.ndarray:
    """Variant C: share of the decisive-feature attribution mass on the recovered channel."""
    live = np.abs(attr[:, DEC_IDX]).sum(1)
    rec = np.abs(attr[:, len(FEATURES):]).sum(1)
    return rec / np.maximum(live + rec, 1e-12)


def evidence_recall(attr_base: np.ndarray, keys: list[dict]) -> np.ndarray:
    """N1's strict recall@|G| with G = the planted decisive features still visible in
    this version; NaN where nothing is visible (all evidence deleted and lost)."""
    out = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        G = set(k["visible_idx"])
        if G:
            out[i] = recall_at_k(attr_base[i], G, len(G))
    return out
