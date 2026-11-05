"""Scoring the four ways of deciding which recovered records to report.

  strict       classic structural carving: intact, consistent cells only (no ML)
  heuristic    fixed share-of-checks score, reported at >= 0.5 (no training)
  ranker_raw   the trained ranker's own scores, reported at >= 0.5
  ranker_cal   the same scores after isotonic calibration, reported at >= 0.5 and
               at the 95%-precision threshold fixed on the calibration split
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ranker import matrix
from .truth import brier, ece, operating_point, ranking


def add_scores(df: pd.DataFrame, bundle: dict, method: str = "isotonic") -> pd.DataFrame:
    X = matrix(df, bundle["features"])
    out = df.copy()
    out["p_raw"] = bundle["model"].predict_proba(X)[:, 1]
    out["p"] = bundle["calibrators"][method].predict_proba(X)[:, 1]
    out["strict_score"] = out["strict"].astype(float)
    return out


def method_table(df: pd.DataFrame, n_deleted: int, n_present: int, t95: float, bins: int) -> pd.DataFrame:
    """One row per method / operating point."""
    specs = [("strict", "strict_score", 1.0), ("heuristic", "heuristic", 0.5),
             ("ranker_raw", "p_raw", 0.5), ("ranker_cal", "p", 0.5), ("ranker_cal_t95", "p", t95)]
    rows = []
    for name, col, thr in specs:
        r = {"method": name, "threshold": thr}
        r.update(operating_point(df, col, thr, n_deleted, n_present))
        y, s = df["y"].to_numpy(), df[col].to_numpy()
        r["ece"] = ece(y, s, bins) if name != "strict" else float("nan")
        r["brier"] = brier(y, s) if name != "strict" else float("nan")
        r["auc"], r["ap"] = ranking(y, s) if name != "strict" else (float("nan"), float("nan"))
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_by_database(df: pd.DataFrame, stats: pd.DataFrame, col: str, thr: float, bins: int,
                          n_boot: int, seed: int) -> dict:
    """95% intervals of precision, recall (present rows) and ECE, resampling whole
    databases with replacement (records of one database are not independent)."""
    keys = ["case_id", "db"]
    dbs = stats[keys + ["present"]].reset_index(drop=True)
    index = {tuple(k): i for i, k in enumerate(dbs[keys].itertuples(index=False))}
    d = df.assign(_db=[index[(c, b)] for c, b in zip(df["case_id"], df["db"])])
    n = len(dbs)
    rep = (d[col] >= thr).to_numpy()
    y, pr, sc = d["y"].to_numpy(), d["present"].to_numpy(), d[col].to_numpy()
    b = np.clip(np.digitize(sc, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    per = np.zeros((n, 3))                                    # reported, tp, tp on present rows
    np.add.at(per, d["_db"].to_numpy(), np.c_[rep, rep * y, rep * y * pr])
    cal = np.zeros((n, bins, 3))                              # count, sum y, sum score
    np.add.at(cal, (d["_db"].to_numpy(), b), np.c_[np.ones(len(d)), y, sc])
    present = dbs["present"].to_numpy(float)
    rng = np.random.default_rng(seed)
    out = {"precision": [], "recall_present": [], "ece": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        R, T, TP = w @ per
        C = np.tensordot(w, cal, axes=1)
        m = C[:, 0] > 0
        out["precision"].append(T / R if R else np.nan)
        out["recall_present"].append(TP / max(1.0, w @ present))
        out["ece"].append(float(np.sum(np.abs(C[m, 1] - C[m, 2])) / C[:, 0].sum()))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}
