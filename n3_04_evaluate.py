"""N3 STEP 6 - Final numbers on the TEST databases (built from N1 test cases).

Compares the four ways of choosing what to report (strict carving, fixed heuristic,
ranker, calibrated ranker) on precision, recall and calibration, then breaks the
calibrated ranker down by SQLite condition and by app database, and checks how
much of the deleted PLANTED EVIDENCE comes back (the link to N1 / N5).
Output: results_n3/test_methods.csv, test_by_condition.csv, test_by_database.csv,
        evidence_recovery.csv, reliability_test.csv, test_scores.csv.gz
Run:  python n3_04_evaluate.py      (about 1 minute)
"""
import joblib
import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed, load_json, save_json
from ufn3.common import load_config_n3, path_n3
from ufn3.evaluate import add_scores, bootstrap_by_database, method_table
from ufn3.ranker import candidates, read_records
from ufn3.truth import ece, operating_point, reliability

cfg3 = load_config_n3()
rc = cfg3["ranker"]
data, models, results = (path_n3(cfg3, k) for k in ("data", "models", "results"))
bundle = joblib.load(models / "ranker.joblib")
meta = load_json(models / "ranker.json")
t95, bins = meta["t95"], rc["ece_bins"]
allrec = read_records(data / "records_test.csv.gz")
te = add_scores(candidates(allrec), bundle, rc["calibration"])
dels = pd.read_csv(data / "deleted_test.csv.gz", dtype={"artifact_id": str, "pattern_id": str})
stats = pd.read_csv(data / "dbstats_test.csv")
nd, npres = int(stats.deleted.sum()), int(stats.present.sum())

banner(f"TEST: {len(stats)} databases, {nd} deleted rows ({npres} still present), "
       f"{len(te)} candidates; {int(allrec.live_dup.sum())} live-row look-alikes excluded")
mt = method_table(te, nd, npres, t95, bins)
ci = bootstrap_by_database(te, stats, "p", rc["report_threshold"], bins, 1000, derive_seed(cfg3["seed"], "boot"))
mt.to_csv(results / "test_methods.csv", index=False)
save_json(ci, results / "test_ci_ranker_cal.json")
print(f"  {'method':16s} {'reported':>8s} {'precision':>9s} {'recall(all)':>11s} {'recall(present)':>15s} "
      f"{'ECE':>6s} {'AUC':>6s}")
for _, r in mt.iterrows():
    print(f"  {r.method:16s} {r.reported:8d} {r.precision:9.3f} {r.recall_all:11.3f} {r.recall_present:15.3f} "
          f"{r.ece:6.3f} {r.auc:6.3f}")
print(f"  95% intervals for ranker_cal (resampling databases): precision {ci['precision'][0]:.3f}-"
      f"{ci['precision'][1]:.3f}, recall(present) {ci['recall_present'][0]:.3f}-{ci['recall_present'][1]:.3f}, "
      f"ECE {ci['ece'][0]:.4f}-{ci['ece'][1]:.4f}")
print("  live-row false positives: 0 for every method above (live look-alikes are excluded by construction)")

rows = []
for group in ("condition", "db"):
    for g, d in te.groupby(group):
        s = stats[stats[group] == g]
        for name, col, thr in (("strict", "strict_score", 1.0), ("ranker_cal", "p", rc["report_threshold"])):
            r = {group: g, "method": name, "databases": len(s), "deleted": int(s.deleted.sum()),
                 "present": int(s.present.sum())}
            r.update(operating_point(d, col, thr, int(s.deleted.sum()), int(s.present.sum())))
            r["ece"] = ece(d.y, d[col], bins) if name != "strict" else np.nan
            rows.append(r)
by = pd.DataFrame(rows)
by[by.condition.notna()].drop(columns="db").to_csv(results / "test_by_condition.csv", index=False)
by[by.db.notna()].drop(columns="condition").to_csv(results / "test_by_database.csv", index=False)
banner("By SQLite condition (ranker_cal at p >= 0.5 vs strict carving)")
bc = by[by.condition.notna()]
for c in bc.condition.unique():
    a, b = bc[(bc.condition == c) & (bc.method == "strict")].iloc[0], bc[(bc.condition == c) & (bc.method == "ranker_cal")].iloc[0]
    print(f"  {c:16s} present {a.present / a.deleted:5.1%} of deleted | recall(present): strict {a.recall_present:.3f}"
          f"  ranker {b.recall_present:.3f} | precision: strict {a.precision:.3f}  ranker {b.precision:.3f}")

# ---------------- planted evidence (N1 role = evidence): artifact level
rep = te[(te.p >= rc["report_threshold"]) & (te.y == 1)]
strict_rep = te[(te.strict == 1) & (te.y == 1)]
ev = dels[dels.role == "evidence"]
art = ev.groupby(["case_id", "artifact_id", "pattern_id", "condition"], as_index=False)["present"].max()
art["recovered"] = art.artifact_id.isin(set(rep.artifact_id)).astype(int)
art["recovered_strict"] = art.artifact_id.isin(set(strict_rep.artifact_id)).astype(int)
evs = art.groupby("pattern_id").agg(deleted=("artifact_id", "size"), present=("present", "mean"),
                                    recovered=("recovered", "mean"),
                                    recovered_strict=("recovered_strict", "mean")).reset_index()
evs.to_csv(results / "evidence_recovery.csv", index=False)
banner(f"Deleted planted evidence: {len(art)} artifacts (any of their rows counts)")
print(f"  bytes still present: {art.present.mean():.1%}   recovered: ranker {art.recovered.mean():.1%}"
      f"  strict {art.recovered_strict.mean():.1%}")
for _, r in evs.iterrows():
    print(f"  {r.pattern_id:18s} {r.deleted:5d} deleted  present {r.present:5.1%}  ranker {r.recovered:5.1%}"
          f"  strict {r.recovered_strict:5.1%}")

rel = [dict(method=m, mean_predicted=a, observed=b, count=c)
       for m, col in (("heuristic", "heuristic"), ("ranker_raw", "p_raw"), ("ranker_cal", "p"))
       for a, b, c in reliability(te.y, te[col], 10)]
pd.DataFrame(rel).to_csv(results / "reliability_test.csv", index=False)
keep = ["case_id", "label", "tier", "db", "condition", "table", "rowid", "values", "y", "present", "artifact_id",
        "role", "pattern_id", "best_hypothesis", "best_source", "strict", "heuristic", "p_raw", "p"]
te[keep].to_csv(results / "test_scores.csv.gz", index=False)
print("\nDone. Next: python n3_05_shift.py")
