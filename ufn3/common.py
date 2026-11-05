"""N3 helpers: config, paths, read-only access to N1's benchmark, value matching."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ufn1.common import ROOT, derive_seed, load_config, load_json

SPLITS = {"train": "train", "calib": "val", "test": "test"}     # N3 split -> N1 split
N1_TYPES = {"sms": "sms", "call": "call", "web": "web", "pay": "txn"}   # N3 database -> N1 artifact type


def load_config_n3() -> dict:
    with open(ROOT / "config_n3.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_n3(cfg3: dict, key: str) -> Path:
    p = ROOT / cfg3["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_n1_cases(split3: str) -> tuple[dict, list[dict], dict[str, pd.DataFrame]]:
    """N1's config, answer key and artifact tables for one split. Clear error if N1 is missing."""
    cfg1 = load_config()
    d = ROOT / cfg1["paths"]["data"]
    s1 = SPLITS[split3]
    need = [d / f"{s1}_answer_key.json"] + [d / f"{s1}_{t}.csv.gz" for t in N1_TYPES.values()]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        raise SystemExit("N1 outputs missing: " + ", ".join(missing) +
                         "\nRun N1 first (python run_all.py).")
    arts = {t: pd.read_csv(d / f"{s1}_{t}.csv.gz", dtype=str, keep_default_na=False)
            for t in N1_TYPES.values()}
    return cfg1, load_json(need[0]), arts


def pick_cases(key: list[dict], n: int, seed: int, split: str) -> list[dict]:
    """n cases stratified by (class, tier); n = 0 keeps every case. Order is fixed."""
    if n <= 0 or n >= len(key):
        return list(key)
    groups: dict[tuple, list[dict]] = {}
    for c in key:
        groups.setdefault((c["label"], c["tier"]), []).append(c)
    rng = np.random.default_rng(derive_seed(seed, "n3_cases", split))
    per = n // len(groups)
    out = []
    for g in sorted(groups):
        items = groups[g]
        idx = rng.choice(len(items), size=min(per, len(items)), replace=False)
        out += [items[i] for i in sorted(idx)]
    return sorted(out, key=lambda c: c["case_id"])


# ------------------------------------------------------------- value matching
def norm_value(v):
    """One canonical, hashable form for comparing a recovered value with ground truth."""
    if v is None:
        return ("null",)
    if isinstance(v, bool):
        return ("int", int(v))
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        if math.isfinite(v) and v == int(v) and abs(v) < 2 ** 53:
            return ("real", float(v))
        return ("real", repr(v))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("blob", bytes(v).hex())
    return ("text", str(v))


def row_key(values: list, ipk: int | None) -> tuple:
    """Content of a row without its rowid-alias column (the rowid is often lost)."""
    return tuple(norm_value(v) for j, v in enumerate(values) if j != ipk)


__all__ = ["load_config_n3", "path_n3", "load_n1_cases", "pick_cases", "norm_value", "row_key",
           "SPLITS", "N1_TYPES", "ROOT"]
