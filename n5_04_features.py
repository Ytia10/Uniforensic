"""STEP 6 - Turn every case version into model inputs for the five variants.

For each split and version: the live artifacts (N1 minus deletions) + the records
N3 recovered (with p and calibrated confidence) -> L, A, Af, B (27 features) and
C (27 + 9 recovered-channel features), plus the version's answer key: the planted
decisive features that are still visible (live or correctly recovered).

The 27 features use N1's scaler (models/scaler.json), so N1's model can read L/A/Af/B
unchanged. The 9 recovered-channel features of C get their own scaler, fit on train.

Output (data_n5/): {split}_raw_{V}.csv.gz, {split}_n5.npz, {split}_answer_key_n5.json,
                   models_n5/rec_scaler.json
Run:  python n5_04_features.py
"""
import time

import numpy as np
import pandas as pd

from ufn1.common import ROOT, banner, load_config, load_json, save_json
from ufn1.schema import CLASSES, DECISIVE_FEATURES, FEATURES
from ufn5.common import SPLITS, load_config_n5, load_split, path_n5
from ufn5.views import (REC_FEATURES, build, fit_rec_scaler, read_recovered, scale,
                        without_recovered)

cfg1, cfg5 = load_config(), load_config_n5()
ddir, mdir = path_n5(cfg5, "data"), path_n5(cfg5, "models")
n1_scaler = load_json(ROOT / cfg1["paths"]["models"] / "scaler.json")
recal = load_json(mdir / "recalibration.json")
thr = cfg5["calibration"]["report_threshold"]
rec_scaler = None

for split in SPLITS:                                   # train first: rec scaler is fit on it
    banner(f"Features for split '{split}'")
    t0 = time.time()
    keys, tables = load_split(split)
    kmap = {k["case_id"]: (i, k) for i, k in enumerate(keys)}
    versions = pd.read_csv(ddir / f"{split}_versions.csv")
    deleted = pd.read_csv(ddir / f"{split}_deleted.csv.gz", keep_default_na=False, dtype=str)
    recovered = read_recovered(ddir / f"{split}_recovered.csv.gz")
    F, visible = build(tables, versions, deleted, recovered, recal, thr)
    for v, df in F.items():
        df.to_csv(ddir / f"{split}_raw_{v}.csv.gz", compression="gzip")
    if split == "train":
        rec_scaler = fit_rec_scaler(F["C"])
        save_json(rec_scaler, mdir / "rec_scaler.json")

    case_idx = np.array([kmap[c][0] for c in versions.case_id])
    y = np.array([CLASSES.index(kmap[c][1]["label"]) for c in versions.case_id])
    arrays = {f"X_{v}": scale(v, F[v], n1_scaler, rec_scaler) for v in F}
    for v in ("L", "C"):                                # input with all recovered records removed
        arrays[f"X0_{v}"] = scale(v, without_recovered(v, F["L"]), n1_scaler, rec_scaler)
    np.savez(ddir / f"{split}_n5.npz", y=y, case_idx=case_idx, level=versions.level.to_numpy(),
             foreign=versions.foreign.to_numpy(), **arrays)

    vkeys = []
    for vid, c in zip(versions.version_id, versions.case_id):
        k = kmap[c][1]
        vis = [f for f in k["primary_features"] if visible.at[vid, f]]
        vkeys.append({"version_id": vid, "case_id": c, "label": k["label"], "tier": k["tier"],
                      "primary_features": k["primary_features"], "visible_features": vis,
                      "visible_idx": [FEATURES.index(f) for f in vis]})
    save_json(vkeys, ddir / f"{split}_answer_key_n5.json")

    off = versions.case_id.map(lambda c: kmap[c][1]["label"] != "benign").to_numpy()
    lv = versions.level.to_numpy()
    print(f"  {len(versions)} versions; recovered records per version: "
          f"{len(recovered) / len(versions):.1f}")
    for s in sorted(set(lv[off])):
        m = off & (lv == s)
        n_prim = np.mean([len(vkeys[i]["primary_features"]) for i in np.where(m)[0]])
        n_vis = np.mean([len(vkeys[i]["visible_features"]) for i in np.where(m)[0]])
        rc = F["C"].loc[versions.version_id[m], REC_FEATURES].abs().sum(axis=1).gt(0).mean()
        print(f"    level {s:4.2f}: offence versions {m.sum():4d}  visible decisive features "
              f"{n_vis:.2f} of {n_prim:.2f}  | recovered channel non-zero in {rc:.2f}")
    dlt = (F["A"][DECISIVE_FEATURES] - F["L"][DECISIVE_FEATURES]).abs().sum(axis=1).gt(0).mean()
    print(f"  versions where recovery changes a decisive feature (A vs L): {dlt:.3f}  "
          f"[{time.time() - t0:.0f}s]")
print("\nDone. Next: python n5_05_train_variants.py")
