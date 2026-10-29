"""Score any triage model the N1 way: accuracy AND explanation correctness.

For one model, on N1's 600 test cases:
  macro_f1, per-class F1, accuracy per tier
  oracle_pass_rate  - share of the 450 offence cases where the model is correct and
                      removing the planted evidence drops P(true class) by >= 0.5 (N1 Step 7)
  recall            - mean recall@|G| of each explainer on the oracle-passed cases (N1 Step 9)
  TER               - Trustworthy-Explanation Rate: share of ALL 450 offence cases where the
                      model is right AND the explanation finds >= 2/3 of the planted evidence.
                      One number that combines "right decision" and "right reason"; no case
                      selection, so models with different oracle sets compare fairly.
  proxy_share       - on correctly classified offence cases: share of the top-|G| features that
                      are PROXIES (features the planted artifacts also changed, e.g. total SMS,
                      but not the designated decisive feature). High = the model leans on
                      indirect clues (see the "proxy problem" in N1, Step 9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ufn1.explainers import run_explainer
from ufn1.metrics import recall_at_k
from ufn1.models import mlp_predict_proba
from ufn1.schema import CLASSES


def proxy_fraction(attr: np.ndarray, key: dict) -> float:
    """Share of the top-|G| features that are proxies (affected but not primary)."""
    G = set(key["primary_idx"])
    proxies = set(key["affected_idx"]) - G
    top = np.argsort(-np.abs(attr), kind="stable")[:len(G)]
    return float(np.mean([t in proxies for t in top]))


def score_model(model, te, keys: list[dict], explainers: list[str], ecfg: dict, seed: int,
                oracle_drop: float, good_recall: float, tag: dict | None = None):
    """Return (summary dict, per-case DataFrame)."""
    tag = tag or {}
    y = te["y"]
    proba = mlp_predict_proba(model, te["X"])
    pred = proba.argmax(1)
    tiers = np.array([k["tier"] for k in keys])
    s = {**tag, "macro_f1": float(f1_score(y, pred, average="macro")),
         "accuracy": float((pred == y).mean())}
    for c, f in zip(CLASSES, f1_score(y, pred, average=None, labels=range(len(CLASSES)))):
        s[f"f1_{c}"] = float(f)
    for t in ("easy", "mixed", "subtle"):
        s[f"acc_{t}"] = float((pred[tiers == t] == y[tiers == t]).mean())

    idx = np.array([i for i, k in enumerate(keys) if k["label"] != "benign"])
    p_cf = mlp_predict_proba(model, te["X_cf"][idx])
    correct = pred[idx] == y[idx]
    drop = proba[idx, y[idx]] - p_cf[np.arange(len(idx)), y[idx]]
    passes = correct & (drop >= oracle_drop)
    s["oracle_pass_rate"] = float(passes.mean())

    rows = pd.DataFrame({**{k: v for k, v in tag.items()}, "case_index": idx,
                         "label": [keys[i]["label"] for i in idx], "tier": tiers[idx],
                         "correct": correct, "oracle_pass": passes})
    for e in explainers:
        attr = run_explainer(e, model, te["X"][idx], pred[idx], te["X"][:1], ecfg[e], seed,
                             progress=False)
        rec = np.array([recall_at_k(a, set(keys[i]["primary_idx"]), len(keys[i]["primary_idx"]))
                        for a, i in zip(attr, idx)])
        prox = np.array([proxy_fraction(a, keys[i]) for a, i in zip(attr, idx)])
        rows[f"recall_{e}"] = rec
        rows[f"proxy_{e}"] = prox
        s[f"proxy_share_{e}"] = float(prox[correct].mean()) if correct.any() else float("nan")
        s[f"recall_{e}"] = float(rec[passes].mean()) if passes.any() else float("nan")
        s[f"ter_{e}"] = float((correct & (rec >= good_recall)).mean())
    return s, rows
