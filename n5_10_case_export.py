"""STEP 12 - N4 alignment: provenance and confidence travel in CASE/UCO.

For a few test versions (one per offence class at 100% deletion, one benign device
with foreign residue): export live + recovered artifacts with N4's CASE exporter plus
UCO's RecoveredObjectFacet and ConfidenceFacet (ufn5.case_provenance), then
  1. read the file back and recompute the B and C features from it alone -> they
     must equal the features of step 6 exactly (confidence is stored in whole
     percent, and N5 already rounds to whole percent, so nothing is lost);
  2. validate every file with the official CASE validator (case_utils 0.18.0, CASE 1.5.0).

Output: data_n5/case_jsonld/*.jsonld, results_n5/case_export_check.json
Run:  python n5_10_case_export.py      (about 1-3 min; the validator is slow)
"""
import json
import time

import numpy as np
import pandas as pd

from ufn1.common import banner, load_json, save_json
from ufn1.schema import DECISIVE_FEATURES
from ufn4.common import load_config_n4
from ufn5.calib import apply_isotonic, to_percent
from ufn5.case_provenance import export_with_provenance, read_with_provenance
from ufn5.common import load_config_n5, load_split, path_n5, per_case
from ufn5.devicedb import TABLES
from ufn5.views import NUMERIC, REC_FEATURES, read_recovered
from ufn5.wfeatures import extract_features_weighted

cfg4, cfg5 = load_config_n4(), load_config_n5()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
out_dir = ddir / "case_jsonld"
out_dir.mkdir(exist_ok=True)
recal = load_json(mdir / "recalibration.json")
keys, tables = load_split("test")
cases = per_case(tables)
label = {k["case_id"]: k["label"] for k in keys}
versions = pd.read_csv(ddir / "test_versions.csv")
deleted = pd.read_csv(ddir / "test_deleted.csv.gz", keep_default_na=False, dtype=str)
recovered = read_recovered(ddir / "test_recovered.csv.gz")
raw_B = pd.read_csv(ddir / "test_raw_B.csv.gz", index_col=0)
raw_C = pd.read_csv(ddir / "test_raw_C.csv.gz", index_col=0)

# --- choose: per offence class the 100%-deleted version with most recovered evidence; one residue device
ev = recovered[(recovered.match_role == "evidence")].groupby("version_id").size()
chosen = []
for cls in ("fraud", "harassment", "cybercrime"):
    cand = [v for v in ev.index if label[v.split("|")[0]] == cls and v.split("|")[1] == "1"]
    chosen.append(max(cand, key=lambda v: (ev[v], v)))
fo = recovered[recovered.label == "foreign"].groupby("version_id").size()
chosen.append(max(fo.index, key=lambda v: (fo[v], v)))

banner(f"CASE export with provenance of {len(chosen)} test versions")
checks = []
for vid in chosen:
    cid = vid.split("|")[0]
    gone = set(deleted[(deleted.version_id == vid) & (deleted.owner == "suspect")].artifact_id)
    live = {t: d[~d.artifact_id.isin(gone)] for t, d in cases[cid].items()}
    rec = {}
    for t in TABLES:
        r = recovered[(recovered.version_id == vid) & (recovered.table == t)].copy()
        for c in NUMERIC.get(t, []):
            r[c] = pd.to_numeric(r[c])
        r["confidence_pct"] = np.round(to_percent(apply_isotonic(recal[t], r.p_raw.to_numpy())) * 100).astype(int)
        r["artifact_id"] = [f"{cid}-rec-{t}-{i}" for i in range(len(r))]
        rec[t] = r
    doc = export_with_provenance(vid, live, rec)
    f = out_dir / f"{vid.replace('|', '_')}.jsonld"
    f.write_text(json.dumps(doc, indent=1))
    back = read_with_provenance(json.loads(f.read_text()))
    fB = extract_features_weighted(back, [vid])
    fL = extract_features_weighted({t: d[~d.recovered] for t, d in back.items()}, [vid])
    dB = float((fB.loc[vid] - raw_B.loc[vid]).abs().max())
    rc = (fB[DECISIVE_FEATURES] - fL[DECISIVE_FEATURES]).loc[vid].to_numpy()
    dC = float(np.abs(rc - raw_C.loc[vid, REC_FEATURES].to_numpy()).max())
    n_rec = sum(len(r) for r in rec.values())
    checks.append({"version_id": vid, "label": label[cid], "recovered_objects": n_rec,
                   "max_diff_B": dB, "max_diff_C_recovered_channel": dC})
    print(f"  {vid:16s} {label[cid]:10s} {n_rec:3d} recovered objects  |B diff| {dB:.1e}  |C diff| {dC:.1e}")

banner(f"Official CASE validation ({cfg4['case_export']['case_version']})")
from case_utils.case_validate import validate                 # noqa: E402  (slow import)
for c in checks:
    t0 = time.time()
    r = validate(str(out_dir / f"{c['version_id'].replace('|', '_')}.jsonld"),
                 case_version=cfg4["case_export"]["case_version"])
    c["conforms"] = bool(r.conforms)
    print(f"  {c['version_id']}: conforms = {r.conforms}   ({time.time() - t0:.0f}s)")
    if not r.conforms:
        print(r.text[:3000])
save_json(checks, rdir / "case_export_check.json")
if not all(c["conforms"] and c["max_diff_B"] < 1e-9 and c["max_diff_C_recovered_channel"] < 1e-9
           for c in checks):
    raise SystemExit("\nCASE PROVENANCE CHECK FAILED - see above.")
print("\nCASE PROVENANCE CHECK PASSED: valid CASE 1.5.0, and B/C features from CASE equal step 6.")
print("Next: python n5_11_dp_release.py")
