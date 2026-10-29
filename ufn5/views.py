"""What the triage model sees for one case version, under each N5 variant.

  live      = N1's artifacts of the case minus the ones the user deleted
  recovered = what the teammate's N3 carved back (valid rows, not exact live copies)

  L   live only                                    w = 1
  A   live + recovered with N3 p >= 0.5            w = 1   (N3's report threshold)
  Af  as A, minus records the teammate's fuzzy live filter would drop
  B   live + every recovered record                w = calibrated confidence
  C   [ 27 features of L | 9 decisive features: B - L ]   (live channel | recovered channel)

Ground-truth views (evaluation only, never model input):
  T     live + the recovered records that are really the suspect's (label 'correct')
  T_ne  T without any planted evidence -> the decisive features that still show the
        planted evidence in this version = the version's answer key (N1 Step 4 logic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ufn1.features import apply_scaler
from ufn1.schema import DECISIVE_FEATURES, FEATURES

from .calib import apply_isotonic, to_percent
from .devicedb import TABLES
from .wfeatures import extract_features_weighted

REC_FEATURES = [f"rec_{f}" for f in DECISIVE_FEATURES]
C_FEATURES = FEATURES + REC_FEATURES
VARIANTS_27 = ("L", "A", "Af", "B")
NUMERIC = {"call": ["duration_s"], "txn": ["amount"]}


def read_recovered(path) -> pd.DataFrame:
    r = pd.read_csv(path, keep_default_na=False, dtype=str)
    r["p_raw"] = r.p_raw.astype(float)
    r["fuzzy_live"] = r.fuzzy_live == "True"
    return r


def _live(split_tables, versions, deleted):
    """Per table: N1 rows replicated for every version of their case, minus deletions."""
    dl = deleted[deleted.owner == "suspect"][["version_id", "artifact_id"]].assign(_del=1)
    out = {}
    for t in TABLES:
        df = split_tables[t].merge(versions[["case_id", "version_id"]], on="case_id")
        df = df.merge(dl, on=["version_id", "artifact_id"], how="left")
        df = df[df._del.isna()].drop(columns=["_del"])
        if "counterparty" in df:
            # N1's CSVs parse call numbers like "+9176..." as integers; recovered records carry
            # them as text. One type for both, or the same number would form two groups.
            df["counterparty"] = df["counterparty"].astype(str)
        out[t] = df.assign(case_id=df.version_id, w=1.0)
    return out


def _rec(recovered, recal):
    out = {}
    for t in TABLES:
        r = recovered[recovered.table == t].copy()
        for c in NUMERIC.get(t, []):
            r[c] = pd.to_numeric(r[c])
        r["p_cal"] = to_percent(apply_isotonic(recal[t], r.p_raw.to_numpy())) if len(r) else []
        out[t] = r.assign(case_id=r.version_id, source="recovered")   # N1's reserved provenance column
    return out


def _feat(live, rec, ids, select=None, weight=None, drop_live=None):
    tabs = {}
    for t in TABLES:
        lv = live[t] if drop_live is None else live[t][~drop_live(live[t])]
        r = rec[t] if select is None else rec[t][select(rec[t])]
        r = r.assign(w=1.0 if weight is None else r[weight].astype(float))
        tabs[t] = pd.concat([lv, r], ignore_index=True)
    return extract_features_weighted(tabs, ids)


def build(split_tables, versions, deleted, recovered, recal, thr: float):
    """Raw (unscaled) feature frames per variant, and per version which decisive
    features still show planted evidence (True/False frame)."""
    ids = versions.version_id.tolist()
    live, rec = _live(split_tables, versions, deleted), _rec(recovered, recal)
    F = {"L": _feat(live, rec, ids, select=lambda r: np.zeros(len(r), bool))}
    F["A"] = _feat(live, rec, ids, select=lambda r: r.p_raw >= thr)
    F["Af"] = _feat(live, rec, ids, select=lambda r: (r.p_raw >= thr) & ~r.fuzzy_live)
    F["B"] = _feat(live, rec, ids, weight="p_cal")
    diff = (F["B"][DECISIVE_FEATURES] - F["L"][DECISIVE_FEATURES]).set_axis(REC_FEATURES, axis=1)
    F["C"] = pd.concat([F["L"], diff], axis=1)
    T = _feat(live, rec, ids, select=lambda r: r.label == "correct")
    T_ne = _feat(live, rec, ids, select=lambda r: (r.label == "correct") & (r.match_role != "evidence"),
                 drop_live=lambda d: d.role == "evidence")
    visible = ((T - T_ne).abs() > 1e-9)[DECISIVE_FEATURES]
    return F, visible


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_rec_scaler(rec_raw: pd.DataFrame) -> dict:
    x = slog(rec_raw[REC_FEATURES].to_numpy(float))
    sd = x.std(0)
    sd[sd == 0] = 1.0
    return {"features": REC_FEATURES, "mean": x.mean(0).tolist(), "scale": sd.tolist()}


def scale(variant: str, raw: pd.DataFrame, n1_scaler: dict, rec_scaler: dict | None) -> np.ndarray:
    X = apply_scaler(raw[FEATURES], n1_scaler)
    if variant != "C":
        return X
    r = (slog(raw[REC_FEATURES].to_numpy(float)) - np.array(rec_scaler["mean"])) / np.array(rec_scaler["scale"])
    return np.hstack([X, r.astype(np.float32)])


def without_recovered(variant: str, raw_L: pd.DataFrame) -> pd.DataFrame:
    """The input the same model gets when every recovered record is removed."""
    if variant != "C":
        return raw_L
    zeros = pd.DataFrame(0.0, index=raw_L.index, columns=REC_FEATURES)
    return pd.concat([raw_L, zeros], axis=1)
