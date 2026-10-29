"""STEP 9 - The recovery-dependence flag: does this decision rest on recovered records?

Two ways to raise it, both computable on a REAL case (no ground truth needed):
  counterfactual flag (any variant) - the predicted class changes when every
        recovered record is removed and the same model is run again (exact test).
  attribution flag (variant C only)  - Integrated Gradients puts >= 50% of the
        decisive-feature attribution on the recovered channel (an explanation-
        based shortcut; checked against the exact counterfactual flag).

Scored against the planted ground truth:
  catch rate   - of the benign devices wrongly triaged as offences, the share that
                 carry the flag (split: clean devices / devices with foreign residue),
                 and the share of those false incriminations that recovered records
                 CAUSED (the call would not be made without them). The counterfactual
                 flag catches every caused one by construction - it is an exact test -
                 so its real cost is the workload; the attribution flag's catch rate
                 on caused cases is a genuine measurement.
  workload     - of the CORRECT offence decisions, the share that carry the flag,
                 i.e. how many decisions an examiner must back with recovered data
                 (expected to grow with the deleted share - that is the point)

Output: results_n5/flag.csv, results_n5/flag_attribution_agreement.csv
Run:  python n5_07_flag.py
"""
from itertools import product

import numpy as np
import pandas as pd

from ufn1.common import banner, load_config
from ufn5.common import load_config_n5, path_n5
from ufn5.sets import BENIGN, load

cfg1, cfg5 = load_config(), load_config_n5()
ddir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "results")
TE = load(ddir, "test")
y, lv, fo = TE["y"], TE["level"], TE["foreign"].astype(bool)
share = cfg5["flag"]["attribution_share"]
rows, agree = [], []
models = list(product(cfg5["variants"], cfg5["foreign_rate_train"], cfg5["train_seeds"]))

banner("Recovery-dependence flag")
for v, rho, seed in models:
    p = np.load(rdir / "pred" / f"{v}_rho{rho}_s{seed}.npz")
    pred, pred0 = p["test_proba"].argmax(1), p["test_proba0"].argmax(1)
    flag = pred != pred0
    ben = y == BENIGN
    fi = ben & (pred != BENIGN)
    base = {"variant": v, "rho": rho, "seed": seed}
    for name, m in (("clean", ben & ~fo), ("residue", ben & fo)):
        rows.append({**base, "group": f"false_incrimination_{name}", "level": "all",
                     "n": int((fi & m).sum()), "flag_rate": float(flag[fi & m].mean()) if (fi & m).any() else np.nan})
        rows.append({**base, "group": f"benign_{name}_all", "level": "all", "n": int(m.sum()),
                     "flag_rate": float(flag[m].mean())})
    caused = fi & (pred0 == BENIGN)          # would NOT be an offence call without recovered records
    for name, m in (("clean", ben & ~fo), ("residue", ben & fo)):
        rows.append({**base, "group": f"false_incrimination_{name}_caused_by_recovered", "level": "all",
                     "n": int((caused & m).sum()), "share_of_fi": float(caused[fi & m].mean()) if (fi & m).any() else np.nan,
                     "flag_rate": float(flag[caused & m].mean()) if (caused & m).any() else np.nan})
    for s in cfg5["scenario"]["levels"]:
        m = ~ben & np.isclose(lv, s)
        ok = m & (pred == y)
        rows.append({**base, "group": "offence_correct", "level": s, "n": int(ok.sum()),
                     "flag_rate": float(flag[ok].mean())})
        rows.append({**base, "group": "offence_wrong", "level": s, "n": int((m & ~ok).sum()),
                     "flag_rate": float(flag[m & ~ok].mean()) if (m & ~ok).any() else np.nan})
    if v == "C":
        af = p["test_rec_share"] >= share
        tp, fp = (af & flag).sum(), (af & ~flag).sum()
        fn = (~af & flag).sum()
        agree.append({**base, "agreement": float((af == flag).mean()),
                      "precision_vs_counterfactual": float(tp / max(tp + fp, 1)),
                      "recall_vs_counterfactual": float(tp / max(tp + fn, 1)),
                      "attr_flag_rate": float(af.mean()), "cf_flag_rate": float(flag.mean()),
                      "attr_catch_rate_recovery_caused_fi": float(af[caused].mean()) if caused.any() else np.nan})

df = pd.DataFrame(rows)
df.to_csv(rdir / "flag.csv", index=False)
pd.DataFrame(agree).to_csv(rdir / "flag_attribution_agreement.csv", index=False)
summ = (df[df.group.str.startswith("false_incrimination")]
        .groupby(["variant", "rho", "group"]).agg(n=("n", "mean"), share_of_fi=("share_of_fi", "mean"),
                                                   flag_rate=("flag_rate", "mean")))
print("  false incriminations carrying the flag (mean over seeds):")
print(summ.round(3).to_string())
w = (df[df.group == "offence_correct"].groupby(["variant", "rho", "level"]).flag_rate.mean()
     .unstack("level"))
print("\n  workload: correct offence decisions carrying the flag, by deleted share:")
print(w.round(3).to_string())
if agree:
    a = pd.DataFrame(agree).groupby("rho").mean(numeric_only=True)
    print("\n  attribution flag (C) vs exact counterfactual flag:")
    print(a[["agreement", "precision_vs_counterfactual", "recall_vs_counterfactual",
         "attr_catch_rate_recovery_caused_fi"]].round(3).to_string())
print("\nDone. Next: python n5_08_conformal.py")
