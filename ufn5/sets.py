"""Which case versions form a training, early-stopping, calibration or test set.

Benign cases exist in two versions (clean / with foreign residue). A set with
foreign rate r uses the residue version for a fixed, seeded share r of the benign
cases; the sets are nested (the cases chosen at r = 0.1 are also chosen at 0.25 ...).
Offence cases: train has one version per case; val/test have one per level.
"""
from __future__ import annotations

import numpy as np

from ufn1.common import derive_seed, load_json
from ufn1.schema import CLASSES

from .common import val_halves

BENIGN = CLASSES.index("benign")


def load(ddir, split: str) -> dict:
    z = dict(np.load(ddir / f"{split}_n5.npz", allow_pickle=False))
    z["keys"] = load_json(ddir / f"{split}_answer_key_n5.json")
    z["case_id"] = np.array([k["case_id"] for k in z["keys"]])
    return z


def _chosen(case_ids: np.ndarray, rate: float, seed: int) -> np.ndarray:
    u = np.array([derive_seed(seed, "foreign_rate", c) / 2**32 for c in case_ids])
    return u < rate


def select(z: dict, rate: float, seed: int, level: float | None = None,
           cases: np.ndarray | None = None) -> np.ndarray:
    """Indices of the versions in the set. level=None -> every level (offence)."""
    benign = z["y"] == BENIGN
    want_foreign = _chosen(z["case_id"], rate, seed)
    keep = np.where(benign, z["foreign"].astype(bool) == want_foreign, True)
    if level is not None:
        keep &= benign | np.isclose(z["level"], level)
    if cases is not None:
        keep &= np.isin(z["case_idx"], cases)
    return np.where(keep)[0]


def one_level_per_case(z: dict, idx: np.ndarray, levels: list[float], seed: int) -> np.ndarray:
    """From val/test indices, keep ONE random level per offence case (benign kept):
    an i.i.d.-style sample for conformal calibration."""
    out = []
    for i in idx:
        if z["y"][i] == BENIGN:
            out.append(i)
        else:
            lv = levels[derive_seed(seed, "cal_level", z["case_id"][i]) % len(levels)]
            if np.isclose(z["level"][i], lv):
                out.append(i)
    return np.array(out)


def val_split_cases(keys_val_n1: list[dict]):
    fit, cal = val_halves(keys_val_n1)
    return fit, cal
