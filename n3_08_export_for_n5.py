"""N3 STEP 10 - Hand the recovered evidence, with its confidence, to N5.

N5 will feed recovered artifacts into triage and explanations, weighted by how sure
the recovery is. This step writes, for every split, each recovered record that a
tool would REPORT - by the calibrated ranker (p >= 0.5) or by the fixed heuristic
(score >= 0.5, the way a typical carver reports) - with its calibrated probability p
and its N1 artifact id when it is a real deleted row, plus the confidence profile
per SQLite condition. Wrong recoveries are kept on purpose (artifact_id empty,
correct = 0): N5 needs them, or confidence-aware triage cannot be tested.
Note: train-split scores are in-sample for the ranker and calib-split scores for the
calibrator; only the test split is fully out-of-sample (column `out_of_sample`).
Output: data_n3/recovered_<split>.csv.gz, results_n3/confidence_profile.csv
Run:  python n3_08_export_for_n5.py      (under 1 minute)
"""
import joblib
import numpy as np
import pandas as pd

from ufn1.common import banner
from ufn3.common import N1_TYPES, load_config_n3, path_n3
from ufn3.evaluate import add_scores
from ufn3.ranker import candidates, read_records

cfg3 = load_config_n3()
rc = cfg3["ranker"]
data, models, results = (path_n3(cfg3, k) for k in ("data", "models", "results"))
bundle = joblib.load(models / "ranker.joblib")
cols = ["case_id", "label", "tier", "db", "condition", "table", "rowid", "values", "artifact_id", "role",
        "pattern_id", "best_hypothesis", "best_source", "p", "heuristic", "correct", "reported_ranker",
        "reported_heuristic"]
prof = []
for split in cfg3["cases_per_split"]:
    d = add_scores(candidates(read_records(data / f"records_{split}.csv.gz")), bundle, rc["calibration"])
    d["correct"] = d["y"]
    d["reported_ranker"] = (d.p >= rc["report_threshold"]).astype(int)
    d["reported_heuristic"] = (d.heuristic >= 0.5).astype(int)
    out = d[(d.reported_ranker == 1) | (d.reported_heuristic == 1)].copy()
    out["artifact_type"] = out.db.map(N1_TYPES)
    out["out_of_sample"] = int(split == "test")
    out[cols + ["artifact_type", "out_of_sample"]].to_csv(data / f"recovered_{split}.csv.gz", index=False)
    rep = out[out.reported_ranker == 1]
    stats = pd.read_csv(data / f"dbstats_{split}.csv")
    for cond, g in rep.groupby("condition"):
        s = stats[stats.condition == cond]
        ok, bad = g[g.correct == 1].p, g[g.correct == 0].p
        prof.append({"split": split, "condition": cond, "deleted": int(s.deleted.sum()),
                     "present": int(s.present.sum()), "reported": len(g), "correct": len(ok),
                     "wrong": len(bad), "recovered_share_of_deleted": len(ok) / max(1, s.deleted.sum()),
                     "p_correct_median": float(np.median(ok)) if len(ok) else np.nan,
                     "p_correct_p05": float(np.percentile(ok, 5)) if len(ok) else np.nan,
                     "p_wrong_median": float(np.median(bad)) if len(bad) else np.nan})
    h = out[out.reported_heuristic == 1]
    banner(f"{split}: {len(out)} records -> data_n3/recovered_{split}.csv.gz")
    print(f"  reported by the ranker:    {len(rep):6d}  ({int(rep.correct.sum())} correct, "
          f"{int((1 - rep.correct).sum())} wrong)")
    print(f"  reported by the heuristic: {len(h):6d}  ({int(h.correct.sum())} correct, "
          f"{int((1 - h.correct).sum())} wrong) - the realistic source of wrong recoveries for N5")
pd.DataFrame(prof).to_csv(results / "confidence_profile.csv", index=False)
print("\nDone. Next: python n3_09_make_report.py")
