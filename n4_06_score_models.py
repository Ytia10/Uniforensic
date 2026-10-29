"""N4 STEP 8 - Score every model the N1 way: accuracy AND explanation correctness.

For each of the ~160 models: macro-F1, accuracy per tier, oracle pass rate, evidence
recall@|G| (Integrated Gradients and Feature Ablation, both validated in N1) and the
Trustworthy-Explanation Rate (TER, see ufn4/evaluate.py). Uses N1's 600 test cases,
N1's counterfactuals and N1's answer key, unchanged.
Output: results_n4/model_scores.csv, results_n4/per_case_scores.csv.gz
Run:  python n4_06_score_models.py      (about 2-4 minutes)
"""
import time

import pandas as pd

from ufn1.common import banner, load_json
from ufn4.common import load_config_n4, load_model, load_n1, path_n4
from ufn4.evaluate import score_model

cfg4 = load_config_n4()
cfg1, data, keys = load_n1()
mdir, rdir = path_n4(cfg4, "models"), path_n4(cfg4, "results")
registry = []
for part in ("baselines", "federated", "poisoning"):
    registry += load_json(mdir / f"registry_{part}.json")

banner(f"Scoring {len(registry)} models on N1's test set")
summaries, cases, t0 = [], [], time.time()
for n, r in enumerate(registry):
    tag = {k: v for k, v in r.items() if k not in ("sha256", "train_counts")}
    s, rows = score_model(load_model(cfg1, mdir / r["file"]), data["test"], keys["test"],
                          cfg4["scoring"]["explainers"], cfg1["explainers"], cfg1["seed"],
                          cfg1["oracle"]["min_prob_drop"], cfg4["scoring"]["good_explanation_recall"], tag)
    summaries.append(s)
    cases.append(rows)
    if n % 25 == 0:
        print(f"  {n:3d}/{len(registry)}  {r['file']:40s} macro-F1 {s['macro_f1']:.3f}  "
              f"TER(IG) {s['ter_integrated_gradients']:.3f}   [{time.time() - t0:.0f}s]")
pd.DataFrame(summaries).to_csv(rdir / "model_scores.csv", index=False)
pd.concat(cases).to_csv(rdir / "per_case_scores.csv.gz", index=False, compression="gzip")
print(f"\nDone in {time.time() - t0:.0f}s. Next: python n4_07_make_report.py")
