"""N5b STEP 5 - How well does the ALTERNATIVE N3 (ufn3) work on N1's databases, and is
its confidence calibrated there?  (Same measurements as n5_03, other carver.)

First checks that N5b planted and deleted EXACTLY the same records as N5 (same phones,
only the carver differs), if N5's data_n5/ is present.

Measured on the TEST databases (never used for fitting anything):
  recovery     - share of deleted records N3 gets back, per table / role / level
  record labels- correct / foreign / wrong among recovered records
  calibration  - ECE and Brier of N3's p (target: the record really existed)
  live filter  - share of CORRECTLY recovered deleted records that the teammate's
                 fuzzy live-row filter (evaluator.is_row_match) would have thrown away
                 (kept so variant Af means the same thing as in N5)
Then fits the per-table isotonic correction on val_fit and re-measures on test.

Output: models_n5b/recalibration.json, results_n5b/n3_on_n1_databases.json,
        results_n5b/reliability.csv, results_n5b/same_truth_as_n5.json
Run:  python n5b_03_calibration.py
"""
import numpy as np
import pandas as pd

from ufn1.common import ROOT, banner, load_json, save_json
from ufn5.calib import GENUINE, apply_isotonic, ece, fit_isotonic, to_percent
from ufn5.common import load_split, path_n5, val_halves
from ufn5.devicedb import TABLES
from ufn5b.common import load_config_n5b

cfg5 = load_config_n5b()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
bins = cfg5["calibration"]["ece_bins"]

keys_val, _ = load_split("val")
fit_idx, _ = val_halves(keys_val)
fit_cases = {keys_val[i]["case_id"] for i in fit_idx}
rv = pd.read_csv(ddir / "val_recovered.csv.gz", keep_default_na=False)
rt = pd.read_csv(ddir / "test_recovered.csv.gz", keep_default_na=False)
dt = pd.read_csv(ddir / "test_deleted.csv.gz", keep_default_na=False)
vt = pd.read_csv(ddir / "test_versions.csv")
rv = rv[rv.case_id.isin(fit_cases)]

banner("Same phones as N5? (planted + deleted records must be identical; only the carver differs)")
n5_del = ROOT / "data_n5" / "test_deleted.csv.gz"
same = {"checked": n5_del.exists()}
if n5_del.exists():
    d5 = pd.read_csv(n5_del, keep_default_na=False)
    cols = sorted(set(d5.columns) & set(dt.columns))
    a = d5[cols].sort_values(cols).reset_index(drop=True)
    b = dt[cols].sort_values(cols).reset_index(drop=True)
    same.update({"n5_rows": len(d5), "n5b_rows": len(dt), "columns_compared": cols,
                 "identical": bool(len(a) == len(b) and a.equals(b))})
    print(f"  N5 test deleted records: {len(d5)}   N5b: {len(dt)}   identical: {same['identical']}")
    if not same["identical"]:
        raise SystemExit("  [FAIL] N5b's deleted truth differs from N5's: the comparison would not be fair. "
                         "Check that config_n5b.yaml has the same scenario/seed as config_n5.yaml.")
else:
    print("  data_n5/ not found: skipped (run N5 first for a like-for-like comparison)")
save_json(same, rdir / "same_truth_as_n5.json")
for r in (rv, rt):
    r["genuine"] = r.label.isin(GENUINE).astype(int)
out = {}

banner("Recovery of deleted records by the alternative N3 (test databases)")
got = set(zip(rt.version_id, rt.match_id))
dt["recovered"] = [(v, a) in got for v, a in zip(dt.version_id, dt.artifact_id)]
dt = dt.merge(vt[["version_id", "level"]], on="version_id")
by_table = dt[dt.owner == "suspect"].groupby("table").recovered.mean()
by_role = dt[dt.owner == "suspect"].groupby("role").recovered.mean()
foreign_rate = float(dt[dt.owner == "foreign"].recovered.mean())
ev = dt[(dt.owner == "suspect") & (dt.role == "evidence")]
by_level = ev.groupby("level").recovered.mean()
print("  share of deleted suspect records recovered, per table:")
for t in TABLES:
    print(f"    {t:6s} {by_table.get(t, float('nan')):.3f}")
print(f"  planted evidence: {by_role.get('evidence', float('nan')):.3f}   background: "
      f"{by_role.get('background', float('nan')):.3f}   foreign residue: {foreign_rate:.3f}")
out["recovery_rate_by_table"] = by_table.to_dict()
out["recovery_rate_by_role"] = by_role.to_dict()
out["recovery_rate_foreign"] = foreign_rate
out["evidence_recovery_rate_by_level"] = {str(k): v for k, v in by_level.items()}

banner("What the recovered records are (test)")
lab = rt.label.value_counts()
print("  " + ", ".join(f"{k}: {v}" for k, v in lab.items()))
out["labels_test"] = lab.to_dict()
out["wrong_share"] = float((rt.label == "wrong").mean())
out["p_raw_by_label"] = rt.groupby("label").p_raw.describe().round(4).to_dict(orient="index")
for k, d in out["p_raw_by_label"].items():
    print(f"    {k:8s} N3 p: mean {d['mean']:.3f}  min {d['min']:.3f}  max {d['max']:.3f}")
thr = cfg5["calibration"]["report_threshold"]
out["share_below_report_threshold"] = float((rt.p_raw < thr).mean())

banner("Teammate's fuzzy live-row filter on correctly recovered DELETED records")
corr = rt[rt.label == "correct"]
fz_t = corr.groupby("table").fuzzy_live.mean()
fz_r = corr.groupby("match_role").fuzzy_live.mean()
for t in TABLES:
    print(f"    {t:6s} would be discarded: {fz_t.get(t, float('nan')):.3f}")
print(f"  planted evidence that would be discarded: {fz_r.get('evidence', float('nan')):.3f}")
out["fuzzy_filter_discard_rate_by_table"] = fz_t.to_dict()
out["fuzzy_filter_discard_rate_by_role"] = fz_r.to_dict()
out["fuzzy_filter_discard_rate_all"] = float(corr.fuzzy_live.mean())

banner("Calibration of the alternative N3's p on N1 databases, then isotonic correction")
carv = load_json(rdir / "carver_ufn3.json")
pub = {r["corpus"]: r["ece"] for r in carv["public_corpus_results"]}
cal, rel = {}, []
rt["p_cal"] = np.nan
for t in TABLES:
    a, b = rv[rv.table == t], rt.table == t
    cal[t] = fit_isotonic(a.p_raw.to_numpy(), a.genuine.to_numpy())
    rt.loc[b, "p_cal"] = to_percent(apply_isotonic(cal[t], rt.loc[b, "p_raw"].to_numpy()))
cal_rep = {"ufn3_public_ece": pub}
for name, col in (("raw", "p_raw"), ("recalibrated", "p_cal")):
    p, y = rt[col].to_numpy(), rt.genuine.to_numpy()
    cal_rep[f"ece_{name}"] = ece(p, y, bins)
    cal_rep[f"brier_{name}"] = float(np.mean((p - y) ** 2))
    cal_rep[f"ece_{name}_by_table"] = {t: ece(rt[rt.table == t][col], rt[rt.table == t].genuine, bins)
                                       for t in TABLES}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for k in range(bins):
        m = idx == k
        if m.any():
            rel.append({"p": name, "bin": k, "count": int(m.sum()), "mean_p": float(p[m].mean()),
                        "share_genuine": float(y[m].mean())})
print("  alternative N3 on PUBLIC corpora (n3_07): " + ", ".join(f"{k} ECE {v:.4f}" for k, v in pub.items()))
print(f"  on N1 test databases, raw N3 p:      ECE {cal_rep['ece_raw']:.4f}   Brier {cal_rep['brier_raw']:.4f}")
print(f"  after isotonic (fit on val_fit):     ECE {cal_rep['ece_recalibrated']:.4f}   "
      f"Brier {cal_rep['brier_recalibrated']:.4f}")
print("  calibrated confidence by label: " + ", ".join(
    f"{k} {v:.3f}" for k, v in rt.groupby("label").p_cal.mean().items()))
out["calibration"] = cal_rep
out["p_cal_by_label"] = rt.groupby("label").p_cal.describe().round(4).to_dict(orient="index")
save_json(cal, mdir / "recalibration.json")
save_json(out, rdir / "n3_on_n1_databases.json")
pd.DataFrame(rel).to_csv(rdir / "reliability.csv", index=False)
print("\nDone. Next: python n5b_04_features.py")
