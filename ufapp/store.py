"""Read-only access to everything N1-N5 produced, loaded once and cached.

Nothing here writes into the N1-N5 folders. Each loader raises a clear error that
names the missing file and the step that creates it.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ufn1.common import ROOT, load_config, load_json
from ufn1.models import TriageMLP
from ufn1.schema import CLASSES, FEATURES
from ufn5.common import load_config_n5, path_n5

_lock = threading.RLock()


class MissingOutput(RuntimeError):
    """An N1-N5 output the app needs is not there."""


def need(path: Path, made_by: str) -> Path:
    if not path.exists():
        raise MissingOutput(f"{path.relative_to(ROOT)} is missing - run {made_by} first.")
    return path


# ---------------------------------------------------------------- configs
@lru_cache(maxsize=None)
def cfg1() -> dict:
    return load_config()


@lru_cache(maxsize=None)
def cfg5() -> dict:
    return load_config_n5()


@lru_cache(maxsize=None)
def cfg_yaml(name: str) -> dict:
    with open(ROOT / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def d5() -> Path:
    return ROOT / cfg5()["paths"]["data"]


def m5() -> Path:
    return ROOT / cfg5()["paths"]["models"]


def r5() -> Path:
    return ROOT / cfg5()["paths"]["results"]


# ---------------------------------------------------------------- N1 test cases
@lru_cache(maxsize=None)
def test_split():
    """(answer keys, {case_id: {table: rows}}) of N1's test split."""
    from ufn5.common import load_split, per_case
    with _lock:
        need(ROOT / cfg1()["paths"]["data"] / "test_answer_key.json", "N1 (python run_all.py)")
        keys, tables = load_split("test")
        return keys, per_case(tables)


@lru_cache(maxsize=None)
def key_of() -> dict:
    return {k["case_id"]: k for k in test_split()[0]}


# ---------------------------------------------------------------- N3 / N5 parameters
@lru_cache(maxsize=None)
def n3_classifier():
    from ufn5.n3_adapter import classifier_from_json
    return classifier_from_json(load_json(need(m5() / "n3_classifier.json", "N5 step 3 (n5_01)")))


@lru_cache(maxsize=None)
def recalibration() -> dict:
    return load_json(need(m5() / "recalibration.json", "N5 step 5 (n5_03)"))


@lru_cache(maxsize=None)
def n1_scaler() -> dict:
    return load_json(need(ROOT / cfg1()["paths"]["models"] / "scaler.json", "N1 step 4"))


@lru_cache(maxsize=None)
def rec_scaler() -> dict:
    return load_json(need(m5() / "rec_scaler.json", "N5 step 6 (n5_04)"))


@lru_cache(maxsize=None)
def test_versions() -> pd.DataFrame:
    return pd.read_csv(need(d5() / "test_versions.csv", "N5 step 4 (n5_02)"))


@lru_cache(maxsize=None)
def stored_raw(variant: str) -> pd.DataFrame:
    """N5's stored raw features of every test version (for the consistency check)."""
    return pd.read_csv(need(d5() / f"test_raw_{variant}.csv.gz", "N5 step 6 (n5_04)"), index_col=0)


@lru_cache(maxsize=None)
def npz(split: str) -> dict:
    from ufn5.sets import load
    with _lock:
        need(d5() / f"{split}_n5.npz", "N5 step 6 (n5_04)")
        return load(d5(), split)


# ---------------------------------------------------------------- models
def model_tag(variant: str, rho, seed: int) -> str:
    return f"{variant}_rho{rho}_s{seed}"


def available_models() -> list[dict]:
    out = []
    for v in cfg5()["variants"]:
        for r in cfg5()["foreign_rate_train"]:
            for s in cfg5()["train_seeds"]:
                if (m5() / f"{model_tag(v, r, s)}.pt").exists():
                    out.append({"variant": v, "rho": r, "seed": s, "n_inputs": 36 if v == "C" else 27})
    return out


@lru_cache(maxsize=None)
def model(variant: str, rho: float, seed: int) -> TriageMLP:
    from ufn5.evaluate import load_mlp
    f = need(m5() / f"{model_tag(variant, rho, seed)}.pt", "N5 step 7 (n5_05)")
    return load_mlp(cfg1(), f, 36 if variant == "C" else 27)


@lru_cache(maxsize=None)
def n1_model() -> TriageMLP:
    from ufn5.evaluate import load_mlp
    return load_mlp(cfg1(), need(ROOT / cfg1()["paths"]["models"] / "mlp.pt", "N1 step 6"), len(FEATURES))


@lru_cache(maxsize=None)
def conformal_q(variant: str, rho: float, seed: int) -> tuple[float, float]:
    """The LAC thresholds N5 step 10 calibrated on val_cal for this model."""
    c = pd.read_csv(need(r5() / "conformal.csv", "N5 step 10 (n5_08)"), dtype={"level": str})
    row = c[(c.variant == variant) & np.isclose(c.rho, float(rho)) & (c.seed == seed)].iloc[0]
    return float(row.q_full), float(row.q_live)


@lru_cache(maxsize=None)
def stored_pred(variant: str, rho: float, seed: int) -> dict:
    p = need(r5() / "pred" / f"{model_tag(variant, rho, seed)}.npz", "N5 step 8 (n5_06)")
    return dict(np.load(p))


@lru_cache(maxsize=None)
def acceptance() -> dict:
    """N1 Level 2 settings: agreement threshold and the calibrated tau."""
    a = load_json(need(ROOT / "results" / "acceptance_summary.json", "N1 step 11"))
    return {"tau": float(a["tau"]), "min_agreement": float(cfg1()["acceptance"]["case_min_agreement"]),
            "primary": cfg1()["acceptance"]["primary_explainer"],
            "secondary": cfg1()["acceptance"]["secondary_explainer"],
            "level1_validated": {k: bool(v["validated"]) for k, v in a["level1"].items()}}


def classes() -> list[str]:
    return list(CLASSES)
