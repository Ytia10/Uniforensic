"""N5b STEP 13 - N2 alignment: can agencies publish HOW MUCH their decisions rest on
recovered records, with N2's differential privacy?

The deployed model is the federated variant C of step 11. Each of N4's 8 agencies
explains its own training versions (Integrated Gradients, predicted class) and
shares N2's per-class summary (ufn2.summaries: clip each case to L2 norm C, sum per
predicted class, add Gaussian noise; secure aggregation + distributed DP). From the
released 36-number mean of each class, the receiver reads the 'recovered reliance':
the share of decisive-feature attribution on the recovered channel.
Clipping bound C = N2's rule (90th percentile of explanation norms) on val versions,
which play the public data role N2's shadow pool plays.

Reported: |released reliance - exact reliance| per class and epsilon, and N2's
evidence recall@3 of the released summary (36 values folded to 27 base features).

Output: results_n5b/dp_release.csv
Run:  python n5b_11_dp_release.py
"""
import math

import numpy as np
import pandas as pd

from ufn1.common import ROOT, banner, derive_seed, load_config, load_json
from ufn1.models import mlp_predict_proba
from ufn1.schema import CLASSES, FEATURES, PATTERNS
from ufn2.common import load_config_n2
from ufn2.summaries import block_sums, clip_rows, evidence_recall_at_3, release
from ufn4.common import load_config_n4
from ufn5.common import path_n5
from ufn5b.common import load_config_n5b
from ufn5.evaluate import DEC_IDX, integrated_gradients, load_mlp, to_base
from ufn5.sets import load, select

cfg1, cfg2, cfg4, cfg5 = load_config(), load_config_n2(), load_config_n4(), load_config_n5b()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
sc, SEED, fcfg = cfg2["summary"], cfg5["seed"], cfg5["federated"]
parts = load_json(ROOT / cfg4["paths"]["data"] / "partitions.json")[f"{fcfg['dirichlet_alpha']}|{fcfg['seed']}"]
TR, VA = load(ddir, "train"), load(ddir, "val")
model = load_mlp(cfg1, mdir / f"fed_C_rho0.25_s{fcfg['seed']}.pt", TR["X_C"].shape[1])
base = np.zeros(TR["X_C"].shape[1], dtype=np.float32)
base[27:] = TR["X0_C"][0, 27:]
NC, CRIME = len(CLASSES), [c for c in CLASSES if c != "benign"]
DECISIVE = {CLASSES.index(c): [FEATURES.index(f) for f in PATTERNS[c].values()] for c in CRIME}


def explain(z, idx):
    pred = mlp_predict_proba(model, z["X_C"][idx]).argmax(1)
    return integrated_gradients(model, z["X_C"][idx], pred, base, cfg5["explain"]["n_steps"]), pred


def reliance(mean_vec):
    live, rec = mean_vec[DEC_IDX].sum(), mean_vec[27:].sum()
    return float(rec / max(live + rec, 1e-12))


banner("DP release of recovered reliance (N2 mechanism, federated variant C)")
iva = select(VA, 0.25, SEED)
a_va, _ = explain(VA, iva)
C = float(np.quantile(np.linalg.norm(np.abs(a_va), axis=1), sc["clip_quantile"]))
itr = select(TR, 0.25, SEED)
a_tr, p_tr = explain(TR, itr)
v_tr = clip_rows(a_tr, C)
blocks = []
for agency in parts:
    m = np.isin(TR["case_idx"][itr], agency)
    blocks.append(block_sums(v_tr[m], p_tr[m], NC))
exact = release("secagg", blocks, C, math.inf, sc["delta"], np.random.default_rng(0))
print(f"  clipping bound C = {C:.4f};  exact recovered reliance per class: " +
      ", ".join(f"{CLASSES[c]} {reliance(exact[c]):.3f}" for c in DECISIVE))
rows = []
for eps in sc["epsilons"]:
    reps = sc["noise_repeats"] if math.isfinite(eps) else 1
    for r in range(reps):
        rel = release("secagg_dp", blocks, C, eps, sc["delta"], np.random.default_rng(derive_seed(SEED, "n5dp", eps, r)))
        for c in DECISIVE:
            folded = to_base(np.maximum(rel[c], 0.0)[None, :])[0]
            rows.append({"eps": eps, "repeat": r, "class": CLASSES[c], "exact_reliance": reliance(exact[c]),
                         "released_reliance": reliance(np.maximum(rel[c], 0.0)),
                         "abs_error": abs(reliance(np.maximum(rel[c], 0.0)) - reliance(exact[c])),
                         "evidence_recall_at_3": evidence_recall_at_3(folded, DECISIVE[c])})
df = pd.DataFrame(rows)
df.to_csv(rdir / "dp_release.csv", index=False)
s = df.groupby("eps").agg(abs_error=("abs_error", "mean"), recall3=("evidence_recall_at_3", "mean"))
print("  mean |released - exact| reliance and evidence recall@3, by epsilon (secagg_dp, 8 agencies):")
print(s.round(4).to_string())
print("\nDone. Next: python n5b_12_make_report.py")
