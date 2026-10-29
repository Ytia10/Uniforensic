"""STEP 9 - Score explanations against the planted answer key + measure disagreement.

Per case and explainer (only cases that PASSED the oracle):
  recall@|G|     - of the |G| top features, how many are planted (strict key). Main metric.
  recall@5       - planted features found in the top 5
  precision@5    - share of the top 5 that is planted (strict key)
  precision@5_lenient - same, but features mechanically touched by the planted
                   artifacts (affected set) also count as correct
  sign_gt        - share of planted features with POSITIVE attribution
  mass_gt        - share of total |attribution| on planted features
  top1_hit       - is the single top feature planted?
Plus the disagreement matrix between explainers (FA, RA, SA, SRA @5, RC, PRA).

Output: results/scores_per_case.csv, results/table_groundtruth.csv,
        results/disagreement_<metric>.csv
Run:  python 07_score_explanations.py
"""
import itertools

import numpy as np
import pandas as pd

from ufn1.common import banner, get_path, load_config, load_json
from ufn1.explainers import EXPLAINERS
from ufn1.metrics import (PAIRWISE, bootstrap_ci, mass_on_gt, precision_at_k, recall_at_k,
                          sign_agreement_gt, topk_idx)

cfg = load_config()
seed, K = cfg["seed"], cfg["scoring"]["k_fixed"]
data_dir, res_dir = get_path(cfg, "data"), get_path(cfg, "results")
keys = load_json(data_dir / "test_answer_key.json")
oracle = load_json(res_dir / "oracle_cases.json")
A = np.load(res_dir / "attributions_mlp.npz")
T = np.load(res_dir / "attributions_xgb.npz")
case_index = A["case_index"]
passes = {(o["model"], o["case_index"]): o["passes"] for o in oracle}

# ------------------------------------------------------------ per-case scores
rows = []
sources = [(e, "mlp", A[e]) for e in EXPLAINERS] + [("treeshap_xgb", "xgb", T["treeshap"])]
for name, model, attr in sources:
    for j, i in enumerate(case_index):
        k = keys[i]
        G, Aff = set(k["primary_idx"]), set(k["affected_idx"])
        a = attr[j]
        rows.append({
            "explainer": name, "case_id": k["case_id"], "case_index": int(i), "label": k["label"],
            "tier": k["tier"], "oracle_pass": passes[(model, int(i))], "n_planted": len(G),
            "recall@|G|": recall_at_k(a, G, len(G)), "recall@5": recall_at_k(a, G, K),
            "precision@5": precision_at_k(a, G, K),
            "precision@5_lenient": precision_at_k(a, Aff, K),
            "sign_gt": sign_agreement_gt(a, G), "mass_gt": mass_on_gt(a, G),
            "top1_hit": float(topk_idx(a, 1)[0] in G)})
df = pd.DataFrame(rows)
df.to_csv(res_dir / "scores_per_case.csv", index=False)

# ------------------------------------------------------- explainer x tier table
banner("Ground-truth scores (oracle-passed cases only), mean [95% CI]")
metrics = ["recall@|G|", "recall@5", "precision@5", "precision@5_lenient", "sign_gt", "mass_gt", "top1_hit"]
ev = df[df.oracle_pass]
table = []
for (name, tier), g in ev.groupby(["explainer", "tier"]):
    row = {"explainer": name, "tier": tier, "n_cases": len(g)}
    for m in metrics:
        mu, lo, hi = bootstrap_ci(g[m].to_numpy(), cfg["scoring"]["bootstrap"], seed)
        row[m], row[m + "_lo"], row[m + "_hi"] = mu, lo, hi
    table.append(row)
for name, g in ev.groupby("explainer"):                  # all tiers pooled
    row = {"explainer": name, "tier": "ALL", "n_cases": len(g)}
    for m in metrics:
        mu, lo, hi = bootstrap_ci(g[m].to_numpy(), cfg["scoring"]["bootstrap"], seed)
        row[m], row[m + "_lo"], row[m + "_hi"] = mu, lo, hi
    table.append(row)
tab = pd.DataFrame(table)
tab.to_csv(res_dir / "table_groundtruth.csv", index=False)
order = EXPLAINERS + ["treeshap_xgb"]
for tier in cfg["benchmark"]["tiers"] + ["ALL"]:
    print(f"\n  tier = {tier}")
    t = tab[tab.tier == tier].set_index("explainer").reindex(order)
    for name, r in t.iterrows():
        print(f"    {name:22s} n={int(r.n_cases):3d}  recall@|G|={r['recall@|G|']:.3f} "
              f"[{r['recall@|G|_lo']:.3f},{r['recall@|G|_hi']:.3f}]  P@5={r['precision@5']:.3f} "
              f"P@5len={r['precision@5_lenient']:.3f}  sign={r['sign_gt']:.3f}  mass={r['mass_gt']:.3f}")

# ------------------------------------------------------- disagreement matrices
banner(f"Explainer disagreement (MLP, oracle-passed cases, k={K})")
mask = np.array([passes[("mlp", int(i))] for i in case_index])
names = [e for e in EXPLAINERS]
for metric, fn in PAIRWISE.items():
    M = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a_name, b_name in itertools.combinations(names, 2):
        vals = [fn(a, b, K) for a, b in zip(A[a_name][mask], A[b_name][mask])]
        M.loc[a_name, b_name] = M.loc[b_name, a_name] = float(np.mean(vals))
    M.to_csv(res_dir / f"disagreement_{metric}.csv")
    if metric in ("FA", "RC"):
        print(f"\n  {metric} (1 = identical, higher = more agreement)")
        print(M.round(2).to_string())
print("\nAll six matrices saved as results/disagreement_<metric>.csv")
print("Done. Next: python 08_ffidelity.py")
