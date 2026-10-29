"""N4 STEP 3 - Prove the shared feature vector can be computed from CASE/UCO.

For 60 test cases (5 per class x tier):
  1. export the case's artifacts to CASE/UCO JSON-LD (one file per case);
  2. read the file back and recompute the 27 features;
  3. compare with N1's features -> must be IDENTICAL;
  4. validate 2 files with the official CASE validator (case_utils, CASE 1.5.0).
Output: data_n4/case_jsonld/<case>.jsonld, results_n4/case_export_check.json
Run:  python n4_01_case_export.py        (about 1-2 minutes)
"""
import json
import time

import pandas as pd

from ufn1.common import banner, get_path, load_json, save_json
from ufn1.features import extract_features
from ufn1.schema import ARTIFACT_TYPES
from ufn4.case_export import from_case_jsonld, to_case_jsonld
from ufn4.common import load_config_n4, load_n1, path_n4

cfg4 = load_config_n4()
cfg1, data, keys = load_n1()
ccfg = cfg4["case_export"]
out_dir = path_n4(cfg4, "data") / "case_jsonld"
out_dir.mkdir(parents=True, exist_ok=True)
res_dir = path_n4(cfg4, "results")
d1 = get_path(cfg1, "data")

# pick the first N cases of every class x tier (deterministic)
chosen = []
for c in cfg1["benchmark"]["classes"]:
    for t in cfg1["benchmark"]["tiers"]:
        chosen += [k["case_id"] for k in keys["test"] if k["label"] == c and k["tier"] == t][:ccfg["cases_per_class_tier"]]
tables = {t: pd.read_csv(d1 / f"test_{t}.csv.gz", keep_default_na=False) for t in ARTIFACT_TYPES}
tables = {t: df[df.case_id.isin(chosen)] for t, df in tables.items()}
n1_feat = pd.read_csv(d1 / "test_features.csv", index_col="case_id").loc[chosen]

banner(f"Export {len(chosen)} cases to CASE/UCO and read them back")
t0, mismatches, n_objects = time.time(), [], 0
for cid in chosen:
    doc = to_case_jsonld(cid, {t: df[df.case_id == cid] for t, df in tables.items()})
    n_objects += len(doc["@graph"])
    with open(out_dir / f"{cid}.jsonld", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    with open(out_dir / f"{cid}.jsonld", "r", encoding="utf-8") as f:
        back = from_case_jsonld(json.load(f))
    diff = (extract_features(back, [cid]).loc[cid] - n1_feat.loc[cid]).abs().max()
    if diff > 1e-9:
        mismatches.append((cid, float(diff)))
print(f"  {len(chosen)} files, {n_objects} CASE objects, {time.time() - t0:.1f}s")
print(f"  cases whose features differ after the round trip: {len(mismatches)}")
if mismatches:
    print("  ", mismatches[:5])

banner(f"Official CASE validation ({ccfg['case_version']}) of {ccfg['validate_cases']} files")
from case_utils.case_validate import validate                 # noqa: E402  (slow import)
val = []
for cid in chosen[:ccfg["validate_cases"]]:
    t0 = time.time()
    r = validate(str(out_dir / f"{cid}.jsonld"), case_version=ccfg["case_version"])
    val.append({"case_id": cid, "conforms": bool(r.conforms)})
    print(f"  {cid}: conforms = {r.conforms}   ({time.time() - t0:.0f}s)")
    if not r.conforms:
        print(r.text[:2000])

save_json({"cases": len(chosen), "objects": n_objects, "feature_mismatches": mismatches,
           "validation": val, "case_version": ccfg["case_version"]}, res_dir / "case_export_check.json")
if mismatches or not all(v["conforms"] for v in val):
    raise SystemExit("\nCASE CHECK FAILED - see above.")
print("\nCASE CHECK PASSED: valid CASE, and features from CASE equal N1's features exactly.")
print("Next: python n4_02_partition.py")
