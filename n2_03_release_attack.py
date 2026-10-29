"""N2 STEP 5 - The real deployment: what do the agencies' shared summaries leak,
and what does protecting them cost in explanation correctness?

For every N4 seed, the 8 agencies of the deployed federation (FedAvg, alpha 0.5)
share per-class explanation summaries under each scheme (ufn2/summaries.py) and
privacy level epsilon. Then:
  PRIVACY  - tracing attack (Dwork et al. 2015): for each agency case (member) and
             each never-seen test case (non-member), can the attacker tell from the
             released summary whether the case is in it? -> AUC, TPR at 1% FPR.
             Non-members are tested against an agency that holds their class, chosen
             in proportion to how many such cases each agency has.
  COST     - evidence recall@3 of the released summaries: for each crime class, do
             its 3 largest values sit on the 3 planted decisive features (N1 key)?
             Per-agency view (none, dp) and global view (secagg, secagg_dp).
Output: results_n2/release_attack.csv
Run:  python n2_03_release_attack.py      (about 2-4 minutes)
"""
import math
import time

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed
from ufn1.schema import CLASSES, FEATURES, PATTERNS
from ufn2.attacks import auc, tpr_at_fpr
from ufn2.common import load_config_n2, load_n1, load_n4, path_n2
from ufn2.summaries import block_sums, clip_rows, evidence_recall_at_3, release

cfg2 = load_config_n2()
sc = cfg2["summary"]
cfg1, data, keys = load_n1()
cfg4, parts, mdir = load_n4()
alpha = cfg2["deployed"]["alpha"]
K, NC = cfg4["agencies"], len(CLASSES)
CRIME = [c for c in CLASSES if c != "benign"]
DECISIVE = {CLASSES.index(c): [FEATURES.index(f) for f in PATTERNS[c].values()] for c in CRIME}
rows, t0 = [], time.time()

for seed in cfg4["run_seeds"]:
    A = np.load(path_n2(cfg2, "data") / f"attr_s{seed}.npz")
    C = float(A["C"])
    v_tr, c_tr = clip_rows(A["attr_train"], C), A["pred_train"]
    v_te, c_te = clip_rows(A["attr_test"], C), A["pred_test"]
    v_sh, c_sh = clip_rows(A["attr_shadow"], C), A["pred_shadow"]
    w = np.array([v_sh[c_sh == c].mean(0) for c in range(NC)])      # attacker's reference means
    agencies = [np.array(i) for i in parts[f"{alpha}|{seed}"]]
    owner = np.empty(len(c_tr), dtype=int)
    for a, idx in enumerate(agencies):
        owner[idx] = a
    blocks = [block_sums(v_tr[idx], c_tr[idx], NC) for idx in agencies]
    counts = np.array([n for _, n in blocks])                       # agencies x classes
    # non-members: claimed agency drawn among agencies holding that class
    rng0 = np.random.default_rng(derive_seed(seed, "claim"))
    claim = np.array([rng0.choice(K, p=counts[:, c] / counts[:, c].sum()) for c in c_te])
    banner(f"Seed {seed}: C = {C:.3f}; cases per agency-class block: "
           f"min {int(counts[counts > 0].min())}, median {int(np.median(counts[counts > 0]))}, max {int(counts.max())}")

    clean = {s: release(s, blocks, C, math.inf, sc["delta"], rng0) for s in ("none", "secagg")}
    for scheme in sc["schemes"]:
        for eps in (sc["epsilons"] if scheme in ("dp", "secagg_dp") else [math.inf]):
            reps = sc["noise_repeats"] if math.isfinite(eps) else 1
            for r in range(reps):
                rng = np.random.default_rng(derive_seed(seed, scheme, eps, r))
                rel = release(scheme, blocks, C, eps, sc["delta"], rng)
                per_agency = scheme in ("none", "dp")
                q_in = np.array([rel[o][c] if per_agency else rel[c] for o, c in zip(owner, c_tr)])
                q_out = np.array([rel[o][c] if per_agency else rel[c] for o, c in zip(claim, c_te)])
                s_in = np.einsum("ij,ij->i", v_tr - w[c_tr], q_in - w[c_tr])
                s_out = np.einsum("ij,ij->i", v_te - w[c_te], q_out - w[c_te])
                # cost: evidence recall@3 of every released crime-class summary
                if per_agency:
                    rec = [evidence_recall_at_3(rel[a][c], DECISIVE[c]) for a in range(K)
                           for c in DECISIVE if counts[a, c] > 0]
                    base = [evidence_recall_at_3(clean["none"][a][c], DECISIVE[c]) for a in range(K)
                            for c in DECISIVE if counts[a, c] > 0]
                else:
                    rec = [evidence_recall_at_3(rel[c], DECISIVE[c]) for c in DECISIVE]
                    base = [evidence_recall_at_3(clean["secagg"][c], DECISIVE[c]) for c in DECISIVE]
                rows.append({"seed": seed, "scheme": scheme, "eps": eps, "repeat": r, "C": C,
                             "auc": auc(s_in, s_out),
                             "tpr_at_1pct_fpr": tpr_at_fpr(s_in, s_out, cfg2["case_mia"]["tpr_at_fpr"]),
                             "evidence_recall3": float(np.mean(rec)),
                             "evidence_recall3_clean": float(np.mean(base)), "n_summaries": len(rec)})
            g = pd.DataFrame([x for x in rows if x["seed"] == seed and x["scheme"] == scheme and x["eps"] == eps])
            print(f"  {scheme:10s} eps {eps:>4}: attack AUC {g.auc.mean():.3f}  TPR@1%FPR {g.tpr_at_1pct_fpr.mean():.3f}"
                  f"  | evidence recall@3 {g.evidence_recall3.mean():.3f} (no noise {g.evidence_recall3_clean.mean():.3f})"
                  f"   [{time.time() - t0:.0f}s]")
pd.DataFrame(rows).to_csv(path_n2(cfg2, "results") / "release_attack.csv", index=False)
print("\nDone. Next: python n2_04_distinguishability.py")
