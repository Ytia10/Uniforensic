"""N5b STEP 8 - Score every model: right decision, right reason, and false incrimination.

For each trained model (variant x rho x seed) and N1's ORIGINAL model (no retraining,
variants L/A/Af/B only), on N1's test cases at every deletion level and foreign rate:
  macro_f1, accuracy, offence_recall
  false_incrimination - share of benign cases triaged as an offence
  evidence recall     - N1's strict recall@|G| of Integrated Gradients, where G = the
                        planted decisive features still visible in that version
  TER                 - N4's Trustworthy-Explanation Rate: share of offence versions
                        (with visible evidence) that are right AND recall >= 0.66
Also stores, for steps 9-11, every model's probabilities with and without the
recovered records, on val and test.

Output: results_n5b/eval_grid.csv, results_n5b/eval_explain.csv, results_n5b/pred/*.npz
Run:  python n5b_06_evaluate.py      (about 3-6 min)
"""
from itertools import product

import numpy as np
import pandas as pd

from ufn1.common import ROOT, banner, load_config
from ufn1.models import mlp_predict_proba
from ufn5.common import path_n5
from ufn5b.common import load_config_n5b
from ufn5.evaluate import (evidence_recall, integrated_gradients, load_mlp, recovered_share,
                           to_base, triage_scores)
from ufn5.sets import BENIGN, load, select

cfg1, cfg5 = load_config(), load_config_n5b()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
pdir = rdir / "pred"
pdir.mkdir(exist_ok=True)
SEED, levels, rates = cfg5["seed"], cfg5["scenario"]["levels"], cfg5["foreign_rates_test"]
good = cfg1["acceptance"]["good_explanation_recall"]
n_steps = cfg5["explain"]["n_steps"]
D = {s: load(ddir, s) for s in ("val", "test")}
TE = D["test"]


def inputs(v: str, z: dict):
    X = z[f"X_{v}"]
    X0 = z["X0_C"] if v == "C" else z["X0_L"]
    return X, X0


def baseline(v: str, z: dict) -> np.ndarray:
    """Zero (= N1 training mean) for live features; 'nothing recovered' for C's channel."""
    b = np.zeros(z[f"X_{v}"].shape[1], dtype=np.float32)
    if v == "C":
        b[27:] = z["X0_C"][0, 27:]           # the scaled value of a raw 0 (same for every row)
    return b


grid, expl = [], []
models = [(v, r, s) for v, r, s in product(cfg5["variants"], cfg5["foreign_rate_train"],
                                           cfg5["train_seeds"])]
models += [(v, "N1", 2026) for v in ("L", "A", "Af", "B")]
banner(f"Scoring {len(models)} models on {len(TE['y'])} test versions")
for v, rho, seed in models:
    tag = f"{v}_rho{rho}_s{seed}"
    if rho == "N1":
        model = load_mlp(cfg1, ROOT / cfg1["paths"]["models"] / "mlp.pt", 27)
    else:
        model = load_mlp(cfg1, mdir / f"{tag}.pt", TE[f"X_{v}"].shape[1])
    store = {}
    for split, z in D.items():
        X, X0 = inputs(v, z)
        P, P0 = mlp_predict_proba(model, X), mlp_predict_proba(model, X0)
        store[f"{split}_proba"], store[f"{split}_proba0"] = P, P0
    X, _ = inputs(v, TE)
    pred = store["test_proba"].argmax(1)
    attr = integrated_gradients(model, X, pred, baseline(v, TE), n_steps)
    store["test_attr_base"] = to_base(attr)
    if v == "C":
        store["test_rec_share"] = recovered_share(attr)
    np.savez_compressed(pdir / f"{tag}.npz", **store)
    rec = evidence_recall(store["test_attr_base"], TE["keys"])

    y = TE["y"]
    for lv, r in product(levels, rates):
        idx = select(TE, r, SEED, level=lv)
        grid.append({"variant": v, "rho": rho, "seed": seed, "level": lv, "foreign_rate": r,
                     **triage_scores(y[idx], pred[idx])})
    for lv in levels:
        idx = select(TE, 0.0, SEED, level=lv)
        off = idx[(y[idx] != BENIGN) & ~np.isnan(rec[idx])]
        ok = pred[off] == y[off]
        expl.append({"variant": v, "rho": rho, "seed": seed, "level": lv, "n_offence_visible": len(off),
                     "evidence_recall": float(np.nanmean(rec[off][ok])) if ok.any() else np.nan,
                     "ter": float((ok & (rec[off] >= good)).mean()) if len(off) else np.nan})
    # benign cases: clean vs foreign-residue version, over all 150 benign test cases
    ben = np.where(y == BENIGN)[0]
    fo = TE["foreign"][ben].astype(bool)
    fi = pred[ben] != BENIGN
    g = [x for x in grid if x["variant"] == v and x["rho"] == rho and x["seed"] == seed]
    m = pd.DataFrame(g).groupby("foreign_rate").macro_f1.mean()
    print(f"  {tag:16s} macro-F1 (r=0) by level " +
          " ".join(f"{x:.3f}" for x in pd.DataFrame(g).query("foreign_rate == 0").macro_f1) +
          f" | false incrim. clean {fi[~fo].mean():.3f}  residue {fi[fo].mean():.3f}")
    grid.append({"variant": v, "rho": rho, "seed": seed, "level": "benign_all", "foreign_rate": "clean",
                 "false_incrimination": float(fi[~fo].mean()), "n": int((~fo).sum())})
    grid.append({"variant": v, "rho": rho, "seed": seed, "level": "benign_all", "foreign_rate": "residue",
                 "false_incrimination": float(fi[fo].mean()), "n": int(fo.sum())})

pd.DataFrame(grid).to_csv(rdir / "eval_grid.csv", index=False)
pd.DataFrame(expl).to_csv(rdir / "eval_explain.csv", index=False)
print("\nDone. Next: python n5b_07_flag.py")
