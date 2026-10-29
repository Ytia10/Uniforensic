"""N2 STEP 6 - Distinguishability game (Pyrgelis, Troncoso & De Cristofaro, NDSS 2018):
how does leakage depend on GROUP SIZE and on how smart the attacker is?

For each group size m and privacy level epsilon, for each target case t:
  1. the attacker builds practice aggregates of m cases from ITS OWN pool (Step 3),
     half with t inside, half with t replaced by another case, and trains two
     classifiers (logistic regression; random forest with 30 trees, as in the paper);
  2. it is tested on aggregates built from the AGENCIES' real cases, same rule;
  3. success = AUC over the test aggregates; also the tracing score (no training).
Two attackers, as the paper recommends:
  adaptive - practises on aggregates with the SAME noise as the real release
  passive  - practises on clean aggregates (overstates the defence; reported to show it)
A group of m cases of one class is what an agency's class summary contains, so
m = 5..50 matches small agency blocks and m = 300+ matches secure aggregation.
Output: results_n2/game.csv
Run:  python n2_04_distinguishability.py      (about 3-6 minutes)
"""
import math
import time

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed
from ufn2.attacks import auc, game_aggregates, make_classifiers
from ufn2.common import load_config_n2, load_n1, load_n4, path_n2
from ufn2.dp import sigma_for
from ufn2.summaries import SENSITIVITY, clip_rows

cfg2 = load_config_n2()
gc, sc = cfg2["game"], cfg2["summary"]
cfg1, data, keys = load_n1()
cfg4, parts, mdir = load_n4()
seed = cfg4["run_seeds"][cfg2["analysis_seed_index"]]
A = np.load(path_n2(cfg2, "data") / f"attr_s{seed}.npz")
C = float(A["C"])
v_tr, c_tr = clip_rows(A["attr_train"], C), A["pred_train"]
v_sh, c_sh = clip_rows(A["attr_shadow"], C), A["pred_shadow"]
rows, t0 = [], time.time()

banner(f"Distinguishability game (seed {seed}): group sizes {gc['group_sizes']}, "
       f"{gc['targets']} targets each, epsilons {sc['epsilons']}")
for m in gc["group_sizes"]:
    rng_t = np.random.default_rng(derive_seed(seed, "targets", m))
    targets = rng_t.choice(len(v_tr), size=gc["targets"], replace=False)
    for ti, t in enumerate(targets):
        c = c_tr[t]
        real_pool = v_tr[(c_tr == c) & (np.arange(len(v_tr)) != t)]   # agencies' other cases of class c
        att_pool = v_sh[c_sh == c]                                    # attacker's own cases of class c
        if len(real_pool) < m or len(att_pool) < m:
            continue
        w = att_pool.mean(0)
        rng = np.random.default_rng(derive_seed(seed, "game", m, ti))
        Xc, bc = game_aggregates(v_tr[t], att_pool, m, gc["train_games"], C, 0.0, rng)   # clean practice
        for eps in sc["epsilons"]:
            sigma = sigma_for(eps, sc["delta"], SENSITIVITY)
            Xa, ba = (Xc, bc) if sigma == 0 else game_aggregates(v_tr[t], att_pool, m, gc["train_games"], C, sigma, rng)
            Xt, bt = game_aggregates(v_tr[t], real_pool, m, gc["test_games"], C, sigma, rng)
            if bt.min() == bt.max():
                continue
            res = {"tracing": auc((Xt[bt == 1] - w) @ (v_tr[t] - w), (Xt[bt == 0] - w) @ (v_tr[t] - w))}
            for mode, (Xtr, btr) in [("adaptive", (Xa, ba)), ("passive", (Xc, bc))]:
                if mode == "passive" and sigma == 0:
                    continue                                            # identical to adaptive
                if btr.min() == btr.max():
                    continue
                for name, clf in make_classifiers(derive_seed(seed, "clf", m, ti), gc["rf_trees"]).items():
                    clf.fit(Xtr, btr)
                    p = clf.predict_proba(Xt)[:, 1]
                    res[f"{name}_{mode}"] = auc(p[bt == 1], p[bt == 0])
            for attack, a in res.items():
                rows.append({"seed": seed, "m": m, "eps": eps, "target": int(t), "attack": attack, "auc": a})
    g = pd.DataFrame([r for r in rows if r["m"] == m])
    # each attack reported on its own: taking the max over attacks per target would
    # inflate chance-level results (max of noisy AUCs > 0.5)
    for attack in ("tracing", "rf_adaptive"):
        mean = g[g.attack == attack].groupby("eps").auc.mean()
        print(f"  m = {m:3d}  {attack:12s} AUC by epsilon  " +
              "  ".join(f"{e:g}:{v:.3f}" for e, v in mean.items()) + f"   [{time.time() - t0:.0f}s]")
pd.DataFrame(rows).to_csv(path_n2(cfg2, "results") / "game.csv", index=False)
print("\nDone. Next: python n2_05_shadow_models.py")
