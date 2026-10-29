"""N4 helpers: config, paths, and loading N1's outputs (read-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from ufn1.common import ROOT, get_path, load_config, load_json
from ufn1.models import TriageMLP
from ufn1.schema import CLASSES, FEATURES


def load_config_n4() -> dict:
    with open(ROOT / "config_n4.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_n4(cfg4: dict, key: str) -> Path:
    p = ROOT / cfg4["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_n1():
    """Everything N4 needs from N1. Raises a clear error if N1 was not run."""
    cfg1 = load_config()
    d = ROOT / cfg1["paths"]["data"]
    need = [d / "train.npz", d / "val.npz", d / "test.npz", d / "test_answer_key.json",
            ROOT / cfg1["paths"]["models"] / "mlp.pt"]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        raise SystemExit("N1 outputs missing: " + ", ".join(missing) +
                         "\nRun the N1 steps first (python run_all.py --skip-attack).")
    data = {s: np.load(d / f"{s}.npz") for s in ("train", "val", "test")}
    keys = {s: load_json(d / f"{s}_answer_key.json") for s in ("train", "val", "test")}
    return cfg1, data, keys


def new_mlp(cfg1: dict) -> TriageMLP:
    return TriageMLP(len(FEATURES), len(CLASSES), cfg1["mlp"]["hidden"], cfg1["mlp"]["dropout"])


def save_model(model, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_model(cfg1: dict, path: Path) -> TriageMLP:
    m = new_mlp(cfg1)
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


__all__ = ["load_config_n4", "path_n4", "load_n1", "new_mlp", "save_model", "load_model",
           "get_path", "ROOT"]
