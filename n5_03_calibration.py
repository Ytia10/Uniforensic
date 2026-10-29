"""STEP 5 - How well does the teammate's N3 work on N1's databases, and is its
confidence calibrated there?

Measured on the TEST databases (never used for fitting anything):
  recovery     - share of deleted records N3 gets back, per table / role / level
  record labels- correct / foreign / wrong among recovered records
  calibration  - ECE and Brier of N3's p (target: the record really existed)
  live filter  - share of CORRECTLY recovered deleted records that the teammate's
                 fuzzy live-row filter (evaluator.is_row_match) would have thrown away
Then fits the per-table isotonic correction on val_fit and re-measures on test.

Output: models_n5/recalibration.json, results_n5/n3_on_n1_databases.json,
        results_n5/reliability.csv
Run:  python n5_03_calibration.py
"""
import numpy as np
import pandas as pd

from ufn1.common import banner, load_json, save_json
from ufn5.calib import GENUINE, apply_isotonic, ece, fit_isotonic, to_percent
from ufn5.common import load_config_n5, load_split, path_n5, val_halves
from ufn5.devicedb import TABLES

cfg5 = load_config_n5()
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
for r in (rv, rt):
    r["genuine"] = r.label.isin(GENUINE).astype(int)
out = {}

banner("Recovery of deleted records by the teammate's N3 (test databases)")
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

banner("Calibration of N3's p on N1 databases, then isotonic correction")
eq = load_json(rdir / "n3_equivalence.json")
cal, rel = {}, []
rt["p_cal"] = np.nan
for t in TABLES:
    a, b = rv[rv.table == t], rt.table == t
    cal[t] = fit_isotonic(a.p_raw.to_numpy(), a.genuine.to_numpy())
    rt.loc[b, "p_cal"] = to_percent(apply_isotonic(cal[t], rt.loc[b, "p_raw"].to_numpy()))
cal_rep = {"teammate_in_sample_ece": eq["teammate_run_pipeline"]["ece_in_sample"]}
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
print(f"  N3's own report (in-sample, its training data): ECE {cal_rep['teammate_in_sample_ece']:.4f}")
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
print("\nDone. Next: python n5_04_features.py")
