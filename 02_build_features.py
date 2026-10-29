"""STEP 4 - Turn artifacts into case-level feature vectors + finish the answer key.

For every case we extract features TWICE:
  full           - from all artifacts                      -> model input
  counterfactual - from all artifacts EXCEPT planted evidence -> "same case, no evidence"
Features that differ between the two are the case's AFFECTED set (lenient key);
the designated decisive features of the planted patterns are the PRIMARY set
(strict key). The counterfactual is reused by the oracle check (step 7).

Output: data/{split}_features.csv, data/{split}_features_cf.csv,
        data/{split}.npz (scaled X, X_cf, y), models/scaler.json,
        data/{split}_answer_key.json (updated)
Run:  python 02_build_features.py
"""
import numpy as np
import pandas as pd

from ufn1.common import banner, get_path, load_config, load_json, save_json
from ufn1.features import apply_scaler, extract_features, fit_scaler
from ufn1.schema import ARTIFACT_TYPES, CLASSES, FEATURES, PATTERN_TO_FEATURE

cfg = load_config()
data_dir, model_dir = get_path(cfg, "data"), get_path(cfg, "models")
scaler = None

for split in ["train", "val", "test"]:                     # train first: scaler is fit on it
    banner(f"Features for split '{split}'")
    keys = load_json(data_dir / f"{split}_answer_key.json")
    case_ids = [k["case_id"] for k in keys]
    tables = {t: pd.read_csv(data_dir / f"{split}_{t}.csv.gz", keep_default_na=False)
              for t in ARTIFACT_TYPES}
    full = extract_features(tables, case_ids)
    cf = extract_features({t: d[d.role != "evidence"] for t, d in tables.items()}, case_ids)
    full.to_csv(data_dir / f"{split}_features.csv")
    cf.to_csv(data_dir / f"{split}_features_cf.csv")

    diff = (full - cf).abs() > 1e-9
    n_dropped = 0
    for k in keys:
        aff = [f for f in FEATURES if diff.at[k["case_id"], f]]
        missing = set(k["primary_features"]) - set(aff)
        if missing:                       # planted, but did not change its feature
            n_dropped += len(missing)     # -> not evidence the model could see
            k["primary_features"] = [f for f in k["primary_features"] if f in aff]
            k["evidence_patterns"] = [p for p in k["evidence_patterns"]
                                      if PATTERN_TO_FEATURE[p] in aff]
        k["affected_features"] = aff
        k["primary_idx"] = [FEATURES.index(f) for f in k["primary_features"]]
        k["affected_idx"] = [FEATURES.index(f) for f in aff]
    save_json(keys, data_dir / f"{split}_answer_key.json")
    print(f"  planted patterns that did not change their feature (removed from key): {n_dropped}")

    if split == "train":
        scaler = fit_scaler(full)
        save_json(scaler, model_dir / "scaler.json")
    y = np.array([CLASSES.index(k["label"]) for k in keys])
    np.savez(data_dir / f"{split}.npz", X=apply_scaler(full, scaler),
             X_cf=apply_scaler(cf, scaler), y=y)

    # quick look: mean of decisive features per class (should differ clearly)
    lab = pd.Series([k["label"] for k in keys], index=full.index)
    show = ["sms_scam_kw_out", "call_max_short_out_single_number", "web_hacktool_visits"]
    print(full[show].groupby(lab).mean().round(2).to_string())
    n_aff = np.mean([len(k["affected_features"]) for k in keys if k["label"] != "benign"])
    print(f"  mean |affected set| for offence cases = {n_aff:.2f}  (primary set = 2 or 3)")

print("\nDone. Next: python 03_sanity_check.py")
