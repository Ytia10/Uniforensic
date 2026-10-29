"""N5 helpers: config, paths, and read-only access to N1's benchmark."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ufn1.common import ROOT, load_config, load_json
from ufn1.schema import ARTIFACT_TYPES, CLASSES

SPLITS = ("train", "val", "test")


def load_config_n5() -> dict:
    with open(ROOT / "config_n5.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_n5(cfg5: dict, key: str) -> Path:
    p = ROOT / cfg5["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_split(split: str):
    """N1 artifacts and answer key of one split (read-only)."""
    cfg1 = load_config()
    d = ROOT / cfg1["paths"]["data"]
    keys = load_json(d / f"{split}_answer_key.json")
    tables = {t: pd.read_csv(d / f"{split}_{t}.csv.gz", keep_default_na=False) for t in ARTIFACT_TYPES}
    return keys, tables


def per_case(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """{case_id: {table: rows of that case}} without copying the data repeatedly."""
    groups = {t: dict(tuple(df.groupby("case_id", sort=False))) for t, df in tables.items()}
    ids = set().union(*[g.keys() for g in groups.values()])
    empty = {t: df.iloc[0:0] for t, df in tables.items()}
    return {c: {t: groups[t].get(c, empty[t]) for t in tables} for c in ids}


def val_halves(keys: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Split N1's validation cases into val_fit (early stopping, recalibration) and
    val_cal (conformal calibration only), stratified by class x tier (alternate cases)."""
    fit, cal = [], []
    seen: dict[tuple, int] = {}
    for i, k in enumerate(keys):
        g = (k["label"], k["tier"])
        (fit if seen.get(g, 0) % 2 == 0 else cal).append(i)
        seen[g] = seen.get(g, 0) + 1
    return np.array(fit), np.array(cal)


def version_id(case_id: str, level: float, foreign: bool) -> str:
    return f"{case_id}|{level:g}|{int(foreign)}"


__all__ = ["ROOT", "SPLITS", "CLASSES", "load_config_n5", "path_n5", "load_split", "per_case",
           "val_halves", "version_id"]
