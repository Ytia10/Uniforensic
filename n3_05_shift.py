"""N3 STEP 7 - Does the ranker still work on settings it never saw in training?

Two hold-out tests on the synthetic databases (nothing else changes):
  leave-one-condition-out  train + calibrate WITHOUT one SQLite condition (e.g. WAL),
                           test only on that condition
  leave-one-app-out        train + calibrate WITHOUT one app database (e.g. payments),
                           test only on that app - its table schemas were never seen
The in-distribution model of step 5 is shown next to each result for comparison.
Output: results_n3/shift_holdout.csv
Run:  python n3_05_shift.py      (about 3-6 minutes)
"""
import joblib
import numpy as np
import pandas as pd

from ufn1.common import banner
from ufn3.common import load_config_n3, path_n3
from ufn3.evaluate import add_scores
from ufn3.ranker import calibrate, candidates, fit_ranker, matrix, read_records
from ufn3.truth import ece, operating_point, ranking

cfg3 = load_config_n3()
rc = cfg3["ranker"]
data, models, results = (path_n3(cfg3, k) for k in ("data", "models", "results"))
tr, ca, te = (candidates(read_records(data / f"records_{s}.csv.gz")) for s in ("train", "calib", "test"))
stats = pd.read_csv(data / "dbstats_test.csv")
te = add_scores(te, joblib.load(models / "ranker.joblib"), rc["calibration"])
thr, bins = rc["report_threshold"], rc["ece_bins"]


def score(d, col, s):
    r = operating_point(d, col, thr, int(s.deleted.sum()), int(s.present.sum()))
    r["ece"] = ece(d.y, d[col], bins)
    r["auc"] = ranking(d.y, d[col])[0]
    return r


rows = []
for kind, col in (("condition", "condition"), ("app", "db")):
    banner(f"Leave-one-{kind}-out (train and calibrate without it, test only on it)")
    for g in sorted(te[col].unique()):
        d, s = te[te[col] == g].copy(), stats[stats[col] == g]
        if d.y.sum() == 0:
            print(f"  {g:16s} no recoverable rows in this setting - skipped")
            continue
        m = fit_ranker(tr[tr[col] != g], rc, cfg3["seed"])
        cal = calibrate(m, ca[ca[col] != g], rc["calibration"])
        d["p_hold"] = cal.predict_proba(matrix(d))[:, 1]
        a, b = score(d, "p", s), score(d, "p_hold", s)
        rows.append({"holdout": kind, "value": g, "deleted": int(s.deleted.sum()), "present": int(s.present.sum()),
                     **{f"in_{k}": v for k, v in a.items()}, **{f"out_{k}": v for k, v in b.items()}})
        print(f"  {g:16s} precision {a['precision']:.3f} -> {b['precision']:.3f}   recall(present) "
              f"{a['recall_present']:.3f} -> {b['recall_present']:.3f}   ECE {a['ece']:.4f} -> {b['ece']:.4f}")
pd.DataFrame(rows).to_csv(results / "shift_holdout.csv", index=False)
print("\n  (left of each arrow: model trained on everything; right: model that never saw it)")
print(f"  worst held-out precision {np.nanmin([r['out_precision'] for r in rows]):.3f}, "
      f"worst held-out recall(present) {np.nanmin([r['out_recall_present'] for r in rows]):.3f}")
print("\nDone. Next: python n3_06_get_corpora.py")
