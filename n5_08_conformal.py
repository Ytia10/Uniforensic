"""STEP 10 - Case-level confidence: conformal prediction sets, with and without the
recovered records.

Calibration: val_cal (the half of N1's validation cases never used for training,
early stopping or recalibration), ONE random deletion level per offence case,
clean benign devices (the analyst cannot know the residue rate in advance).
Test: every level x foreign rate, plus a 'mixture' test set drawn the same way as
calibration (one level per case, clean), where the 90% guarantee applies.

Reports coverage and mean set size of the full set and the live set, and the
coverage on benign devices with foreign residue. Cross-checks the full sets of every
model against MAPIE's SplitConformalClassifier.

Output: results_n5/conformal.csv, results_n5/conformal_mapie_check.json
Run:  python n5_08_conformal.py
"""
from itertools import product

import numpy as np
import pandas as pd

from ufn1.common import banner, save_json
from ufn5.common import load_config_n5, load_split, path_n5
from ufn5.conformal import coverage, lac_quantile, lac_sets, mapie_sets
from ufn5.sets import BENIGN, load, one_level_per_case, select, val_split_cases

cfg5 = load_config_n5()
ddir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "results")
SEED, alpha = cfg5["seed"], cfg5["conformal"]["alpha"]
levels, rates = cfg5["scenario"]["levels"], cfg5["foreign_rates_test"]
VA, TE = load(ddir, "val"), load(ddir, "test")
_, CAL_CASES = val_split_cases(load_split("val")[0])
cal = one_level_per_case(VA, select(VA, 0.0, SEED, cases=CAL_CASES), levels, SEED)
mix = one_level_per_case(TE, select(TE, 0.0, SEED), levels, SEED + 1)
y = TE["y"]
rows, mapie_check = [], []
banner(f"Conformal sets (alpha = {alpha}); calibration n = {len(cal)}, mixture test n = {len(mix)}")

for v, rho, seed in product(cfg5["variants"], cfg5["foreign_rate_train"], cfg5["train_seeds"]):
    p = np.load(rdir / "pred" / f"{v}_rho{rho}_s{seed}.npz")
    q = lac_quantile(p["val_proba"][cal], VA["y"][cal], alpha)
    q0 = lac_quantile(p["val_proba0"][cal], VA["y"][cal], alpha)
    S, S0 = lac_sets(p["test_proba"], q), lac_sets(p["test_proba0"], q0)
    base = {"variant": v, "rho": rho, "seed": seed, "q_full": q, "q_live": q0}

    def rec(level, rate, idx):
        rows.append({**base, "level": level, "foreign_rate": rate, "n": len(idx),
                     "coverage_full": coverage(S[idx], y[idx]), "size_full": float(S[idx].sum(1).mean()),
                     "coverage_live": coverage(S0[idx], y[idx]), "size_live": float(S0[idx].sum(1).mean()),
                     "singleton_full": float((S[idx].sum(1) == 1).mean())})

    rec("mixture", 0.0, mix)
    for lv, r in product(levels, rates):
        rec(lv, r, select(TE, r, SEED, level=lv))
    ben_res = np.where((y == BENIGN) & TE["foreign"].astype(bool))[0]
    rec("benign_residue", 1.0, ben_res)

    # independent cross-check with MAPIE on exactly the same probabilities
    P_all = np.vstack([p["val_proba"], p["test_proba"]])
    M = mapie_sets(P_all, cal, VA["y"][cal], len(VA["y"]) + np.arange(len(y)), alpha)
    mapie_check.append({**base, "identical_sets_share": float((M == S).all(1).mean())})
    r0 = [x for x in rows if x["variant"] == v and x["rho"] == rho and x["seed"] == seed]
    mx, br = r0[0], r0[-1]
    print(f"  {v:3s} rho={rho:<4} s={seed}  mixture: cover {mx['coverage_full']:.3f} size "
          f"{mx['size_full']:.2f} | live {mx['coverage_live']:.3f}/{mx['size_live']:.2f}  || "
          f"benign residue: full {br['coverage_full']:.3f}  live {br['coverage_live']:.3f}  "
          f"| MAPIE identical {mapie_check[-1]['identical_sets_share']:.4f}")

pd.DataFrame(rows).to_csv(rdir / "conformal.csv", index=False)
save_json({"min_identical_share": min(m["identical_sets_share"] for m in mapie_check),
           "per_model": mapie_check}, rdir / "conformal_mapie_check.json")
print("\nDone. Next: python n5_09_federated.py")
