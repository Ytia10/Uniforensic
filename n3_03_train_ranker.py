"""N3 STEP 5 - Train the ranker and calibrate its scores into probabilities.

Training records: the train split (N1 train cases). Calibration: the calib split
(N1 validation cases), never seen by the ranker. Test records are not touched.
Two operating points are fixed here, BEFORE any test result is seen:
  p >= 0.5                  report a record when it is more likely right than wrong
  p >= t95                  the lowest threshold reaching 95% precision on calib
Output: models_n3/ranker.joblib, models_n3/ranker.json, results_n3/feature_importance.csv
Run:  python n3_03_train_ranker.py      (under 1 minute)
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from ufn1.common import banner, save_json
from ufn3.carver import MODEL_FEATURES
from ufn3.common import load_config_n3, path_n3
from ufn3.ranker import calibrate, candidates, fit_ranker, matrix, read_records, threshold_for_precision
from ufn3.truth import ece

cfg3 = load_config_n3()
rc = cfg3["ranker"]
data, models, results = (path_n3(cfg3, k) for k in ("data", "models", "results"))
tr = candidates(read_records(data / "records_train.csv.gz"))
ca = candidates(read_records(data / "records_calib.csv.gz"))
banner(f"Train {len(tr)} candidates ({int(tr.y.sum())} correct) | calibrate on {len(ca)} ({int(ca.y.sum())} correct)")

model = fit_ranker(tr, rc, cfg3["seed"])
print(f"  ranker: {model.n_iter_} boosting rounds (early stopping on 10% of train)")
cal = {m: calibrate(model, ca, m) for m in ("isotonic", "sigmoid")}
raw_ca = model.predict_proba(matrix(ca))[:, 1]
p_ca = cal[rc["calibration"]].predict_proba(matrix(ca))[:, 1]
t95 = threshold_for_precision(ca["y"].to_numpy(), p_ca, rc["target_precision"])
print(f"  calib ECE: raw scores {ece(ca.y, raw_ca, rc['ece_bins']):.3f}  ->  "
      f"{rc['calibration']} {ece(ca.y, p_ca, rc['ece_bins']):.3f}  (in-sample for the calibrator)")
print(f"  operating points: p >= {rc['report_threshold']}  and  p >= {t95:.3f} "
      f"(95% precision on calib)")

pi = permutation_importance(model, matrix(ca), ca["y"].to_numpy(), scoring="average_precision",
                            n_repeats=5, random_state=cfg3["seed"])
imp = pd.DataFrame({"feature": MODEL_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
imp = imp.sort_values("importance", ascending=False)
imp.to_csv(results / "feature_importance.csv", index=False)
print("  most useful features (drop in average precision when shuffled, calib split):")
for _, r in imp.head(8).iterrows():
    print(f"    {r.feature:22s} {r.importance:.3f}")

joblib.dump({"model": model, "calibrators": cal, "features": MODEL_FEATURES}, models / "ranker.joblib")
save_json({"features": MODEL_FEATURES, "calibration": rc["calibration"], "report_threshold": rc["report_threshold"],
           "t95": t95, "target_precision": rc["target_precision"], "n_iter": int(model.n_iter_),
           "sklearn": sklearn.__version__, "train_candidates": len(tr), "calib_candidates": len(ca),
           "train_positive_share": float(np.mean(tr.y))}, models / "ranker.json")
print("\nDone. Next: python n3_04_evaluate.py")
