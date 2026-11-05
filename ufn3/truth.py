"""Ground truth and scoring for recovered records.

Unit of evaluation = one RECOVERED RECORD (unique decoded content per table and
database), because that is what an examiner reads in a report.
  correct     its content equals a deleted row version, column for column
  live FP     its content equals a row that is still live (a tool that reports it
              as "deleted" misleads the examiner) - our methods exclude these by
              construction, external tools are scored on them
  wrong       anything else (garbage, a row with overwritten columns, a wrong guess
              of lost header bytes)
Recall uses two denominators (as in sqlite4n6's validation): ALL deleted rows, and
the PRESENT ones whose value bytes still survive contiguously in the files. A row
whose bytes were overwritten cannot be recovered by any carver.
"""
from __future__ import annotations

import struct

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import row_key


def _encode_value(v) -> bytes:
    """Body bytes SQLite writes for a value (smallest integer size; 0 and 1 take none)."""
    if v is None:
        return b""
    if isinstance(v, int):
        if v in (0, 1):
            return b""
        for n in (1, 2, 3, 4, 6, 8):
            if -(1 << (8 * n - 1)) <= v < (1 << (8 * n - 1)):
                return v.to_bytes(n, "big", signed=True)
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 2 ** 51:        # SQLite stores integral REALs as integers
            return _encode_value(int(v))
        return struct.pack(">d", v)
    if isinstance(v, str):
        return v.encode("utf-8")
    return bytes(v)


def record_body(values: list, ipk: int | None) -> bytes:
    return b"".join(_encode_value(v) for j, v in enumerate(values) if j != ipk)


def truth_sets(gt: dict, blobs: list[bytes]) -> dict:
    """Deleted, live and present keys of one database; a key is (table, row content).
    present = the row's value bytes still occur contiguously in the main file or WAL."""
    deleted, live, present, meta = set(), set(), set(), {}
    for table, t in gt["tables"].items():
        for v in t["versions"]:
            k = (table, row_key(v["values"], t["ipk"]))
            if v["deleted"]:
                deleted.add(k)
                meta[k] = v
                body = record_body(v["values"], t["ipk"])
                if any(body in b for b in blobs):
                    present.add(k)
            else:
                live.add(k)
    return {"deleted": deleted, "live": live, "present": present, "meta": meta}


# ------------------------------------------------------------------ metrics
def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: average |observed rate - mean predicted probability|
    over equal-width probability bins, weighted by bin size (Guo et al., ICML 2017)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def reliability(y, p, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) per bin - the reliability diagram."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
            for b in range(bins) if (idx == b).any()]


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ranking(y, p) -> tuple[float, float]:
    """ROC AUC and average precision (nan if only one class is present)."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def operating_point(df, score: str, thr: float, n_deleted: int, n_present: int) -> dict:
    """Precision and both recalls when every record with score >= thr is reported.
    df has one row per recovered record with columns y (1 correct), present (1 if its
    truth row is present), live_dup (1 if it equals a live row)."""
    rep = df[(df[score] >= thr) & (df["live_dup"] == 0)]
    tp = int(rep["y"].sum())
    return {"reported": int(len(rep)), "tp": tp, "fp": int(len(rep) - tp),
            "precision": tp / len(rep) if len(rep) else float("nan"),
            "recall_all": tp / n_deleted if n_deleted else float("nan"),
            "recall_present": int(rep["y"][rep["present"] == 1].sum()) / n_present if n_present else float("nan")}
