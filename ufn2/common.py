"""N2 helpers: config, paths, and read-only access to N1 and N4 outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ufn1.common import ROOT, load_json
from ufn1.explainers import run_explainer
from ufn1.models import mlp_predict_proba
from ufn4.common import load_config_n4, load_model, load_n1, path_n4


def load_config_n2() -> dict:
    with open(ROOT / "config_n2.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_n2(cfg2: dict, key: str) -> Path:
    p = ROOT / cfg2["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_n4():
    """N4's config, partitions and model folder. Clear error if N4 was not run."""
    cfg4 = load_config_n4()
    need = [path_n4(cfg4, "data") / "partitions.json", path_n4(cfg4, "models") / "registry_federated.json"]
    missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
    if missing:
        raise SystemExit("N4 outputs missing: " + ", ".join(missing) +
                         "\nRun N4 first (python n4_run_all.py).")
    return cfg4, load_json(need[0]), path_n4(cfg4, "models")


def deployed_model(cfg1, cfg2, cfg4, seed: int):
    d = cfg2["deployed"]
    return load_model(cfg1, path_n4(cfg4, "models") / f"fed_{d['method']}_a{d['alpha']}_s{seed}.pt")


CHUNK = 250   # Captum IG time grows ~quadratically with batch size; chunks keep it linear


def explain(model, X: np.ndarray, cfg1: dict, cfg2: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Attributions explaining the model's PREDICTED class, plus the predictions.
    Cases are explained in chunks; each case's attribution does not depend on the
    other cases in its batch, so chunking changes speed, not results."""
    pred = mlp_predict_proba(model, X).argmax(1)
    e = cfg2["explainer"]
    attr = np.vstack([run_explainer(e, model, X[i:i + CHUNK], pred[i:i + CHUNK], X[:1],
                                    cfg1["explainers"][e], seed, progress=False)
                      for i in range(0, len(X), CHUNK)])
    return attr, pred


__all__ = ["load_config_n2", "path_n2", "load_n1", "load_n4", "deployed_model", "explain", "ROOT"]
