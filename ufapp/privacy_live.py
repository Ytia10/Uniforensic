"""N2's differential privacy, live: publish how much decisions rest on recovered records.

Same computation as N5 step 13 (n5_11_dp_release.py), split in two so the epsilon
slider is instant:
  prepare()  - explain every agency's training versions with the federated variant-C
               model and build N2's per-class sums (about 20-40 s, done once)
  release()  - N2's ufn2.summaries.release('secagg_dp', ...) for one epsilon and one
               noise draw, seeded exactly as n5_11 seeds it
The app checks its released numbers against results_n5/dp_release.csv.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from ufn1.common import ROOT, derive_seed, load_config, load_json
from ufn1.models import mlp_predict_proba
from ufn1.schema import CLASSES, FEATURES, PATTERNS
from ufn2.common import load_config_n2
from ufn2.summaries import block_sums, clip_rows, evidence_recall_at_3, release
from ufn4.common import load_config_n4
from ufn5.evaluate import DEC_IDX, integrated_gradients, load_mlp, to_base

from . import store

_cache: dict = {}
_lock = threading.Lock()
CRIME = [c for c in CLASSES if c != "benign"]
DECISIVE = {CLASSES.index(c): [FEATURES.index(f) for f in PATTERNS[c].values()] for c in CRIME}


def _reliance(v: np.ndarray) -> float:
    live, rec = v[DEC_IDX].sum(), v[27:].sum()
    return float(rec / max(live + rec, 1e-12))


def prepare(job) -> dict:
    from ufn5.sets import select
    cfg1, cfg2, cfg4, cfg5 = load_config(), load_config_n2(), load_config_n4(), store.cfg5()
    fcfg, SEED = cfg5["federated"], cfg5["seed"]
    job.step("Load the federated variant-C model (N5 step 11) and N4's 8-agency partition")
    parts = load_json(ROOT / cfg4["paths"]["data"] / "partitions.json")[f"{fcfg['dirichlet_alpha']}|{fcfg['seed']}"]
    TR, VA = store.npz("train"), store.npz("val")
    model = load_mlp(cfg1, store.need(store.m5() / f"fed_C_rho0.25_s{fcfg['seed']}.pt", "N5 step 11 (n5_09)"),
                     TR["X_C"].shape[1])
    base = np.zeros(TR["X_C"].shape[1], dtype=np.float32)
    base[27:] = TR["X0_C"][0, 27:]
    job.done_step(f"{len(parts)} agencies")

    def explain(z, idx):
        pred = mlp_predict_proba(model, z["X_C"][idx]).argmax(1)
        return integrated_gradients(model, z["X_C"][idx], pred, base, cfg5["explain"]["n_steps"]), pred

    job.step("Explain the public stand-in (val) to set N2's clipping bound C")
    a_va, _ = explain(VA, select(VA, 0.25, SEED))
    C = float(np.quantile(np.linalg.norm(np.abs(a_va), axis=1), cfg2["summary"]["clip_quantile"]))
    job.done_step(f"C = {C:.4f}")
    job.step("Each agency explains its own cases and builds N2's per-class sums")
    itr = select(TR, 0.25, SEED)
    a_tr, p_tr = explain(TR, itr)
    v_tr = clip_rows(a_tr, C)
    blocks = [block_sums(v_tr[np.isin(TR["case_idx"][itr], ag)], p_tr[np.isin(TR["case_idx"][itr], ag)], len(CLASSES))
              for ag in parts]
    exact = release("secagg", blocks, C, math.inf, cfg2["summary"]["delta"], np.random.default_rng(0))
    job.done_step(", ".join(f"{CLASSES[c]} {_reliance(exact[c]):.3f}" for c in DECISIVE))
    with _lock:
        _cache.update(blocks=blocks, C=C, exact=exact, delta=cfg2["summary"]["delta"], seed=SEED,
                      epsilons=cfg2["summary"]["epsilons"], repeats=cfg2["summary"]["noise_repeats"])
    return {"C": C, "exact": {CLASSES[c]: _reliance(exact[c]) for c in DECISIVE},
            "epsilons": [None if not math.isfinite(e) else e for e in cfg2["summary"]["epsilons"]],
            "repeats": cfg2["summary"]["noise_repeats"]}


def ready() -> bool:
    return "blocks" in _cache


def do_release(eps: float, repeat: int) -> dict:
    with _lock:
        c = dict(_cache)
    if not c:
        raise RuntimeError("prepare the privacy demo first")
    match = [e for e in c["epsilons"] if float(e) == float(eps)]
    if not match:
        raise ValueError(f"epsilon must be one of N2's values {c['epsilons']}")
    eps = match[0]                       # the config's own value: the seed hashes its text
    rel = release("secagg_dp", c["blocks"], c["C"], eps, c["delta"],
                  np.random.default_rng(derive_seed(c["seed"], "n5dp", eps, repeat)))
    rows = []
    for k in DECISIVE:
        v = np.maximum(rel[k], 0.0)
        rows.append({"class": CLASSES[k], "exact_reliance": _reliance(c["exact"][k]),
                     "released_reliance": _reliance(v),
                     "abs_error": abs(_reliance(v) - _reliance(c["exact"][k])),
                     "evidence_recall_at_3": evidence_recall_at_3(to_base(v[None, :])[0], DECISIVE[k]),
                     "released_vector": v.round(5).tolist()})
    return {"eps": None if not math.isfinite(eps) else eps, "repeat": repeat, "rows": rows}
