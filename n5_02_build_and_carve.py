"""STEP 4 - Put every N1 case on a simulated phone, delete, carve with the teammate's N3.

For every case version (case x deletion level x foreign residue yes/no):
  plan deletions (ufn5.scenario) -> build 5 real SQLite databases in time order
  (ufn5.devicedb) -> carve each with the teammate's SQLiteCarver + classifier
  (ufn5.n3_adapter) -> label each recovered record correct / foreign / wrong.

Versions:
  offence cases, val + test : every level in config_n5 scenario.levels
  offence cases, train      : ONE level per case, drawn at random (seeded)
  benign cases, every split : clean and with foreign residue. A benign case has no
                              planted evidence, so every level gives the SAME
                              database; it is built once and reused for all levels.

Output (data_n5/): {split}_versions.csv, {split}_deleted.csv.gz (truth of what was
deleted, incl. foreign residue), {split}_recovered.csv.gz (what N3 got back),
sample_dbs/ (the databases of 2 test cases, to open in any SQLite browser).
Run:  python n5_02_build_and_carve.py        (about 15-30 min on 2 CPU cores)
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed, load_config
from ufn5.common import SPLITS, load_config_n5, load_split, path_n5, per_case, version_id
from ufn5.n3_adapter import classifier_from_json
from ufn5.recover import run_case
from ufn1.common import load_json

cfg1, cfg5 = load_config(), load_config_n5()
ddir, mdir = path_n5(cfg5, "data"), path_n5(cfg5, "models")
scfg, seed, levels = cfg5["scenario"], cfg5["seed"], cfg5["scenario"]["levels"]
window = cfg1["benchmark"]["window_days"]
CLF = classifier_from_json(load_json(mdir / "n3_classifier.json"))
SAMPLE_DIR = ddir / "sample_dbs"
SAMPLE_DIR.mkdir(exist_ok=True)


def work(task):
    case_id, ctab, level, foreign, keep = task
    keep_dir = None
    if keep:
        keep_dir = SAMPLE_DIR / version_id(case_id, level, foreign).replace("|", "_")
        keep_dir.mkdir(exist_ok=True)
        keep_dir = str(keep_dir)
    truth, rec = run_case(case_id, ctab, level, foreign, CLF, scfg, seed, window, keep_dir)
    vid = version_id(case_id, level, foreign)
    dele = truth[truth.deleted].drop(columns=["deleted"]).assign(version_id=vid)
    rec = rec.assign(version_id=vid)
    summary = {"version_id": vid, "case_id": case_id, "level": level, "foreign": foreign,
               "n_artifacts": int((truth.owner == "suspect").sum()),
               "n_deleted": int(((truth.owner == "suspect") & truth.deleted).sum()),
               "n_evidence": int((truth.role == "evidence").sum()),
               "n_evidence_deleted": int(((truth.role == "evidence") & truth.deleted).sum()),
               "n_foreign": int((truth.owner == "foreign").sum()),
               "n_recovered": len(rec),
               "behaviour": truth.loc[truth.owner == "suspect", "behaviour"].iloc[0]}
    return summary, dele, rec


def tasks_for(split, keys, cases):
    sample = {k["case_id"] for k in keys[::150][:2]} if split == "test" else set()
    for k in keys:
        cid, ctab = k["case_id"], cases[k["case_id"]]
        if k["label"] == "benign":
            for foreign in (False, True):
                yield cid, ctab, 0.0, foreign, cid in sample
        elif split == "train":
            lv = levels[derive_seed(seed, "train_level", cid) % len(levels)]
            yield cid, ctab, float(lv), False, False
        else:
            for lv in levels:
                yield cid, ctab, float(lv), False, cid in sample and lv == 0.5


if __name__ == "__main__":
    for split in SPLITS:
        banner(f"Split '{split}': build databases, delete, carve with N3")
        t0 = time.time()
        keys, tables = load_split(split)
        cases = per_case(tables)
        tasks = list(tasks_for(split, keys, cases))
        print(f"  {len(tasks)} case versions x 5 databases, {cfg5['workers']} worker(s)")
        summ, dele, rec = [], [], []
        with Pool(cfg5["workers"]) as pool:
            for i, (s, d, r) in enumerate(pool.imap(work, tasks, chunksize=8)):
                summ.append(s), dele.append(d), rec.append(r)
                if (i + 1) % 500 == 0:
                    print(f"    {i + 1}/{len(tasks)}  ({time.time() - t0:.0f}s)")
        summ = pd.DataFrame(summ)
        rec = pd.concat(rec, ignore_index=True)
        pd.concat(dele, ignore_index=True).to_csv(ddir / f"{split}_deleted.csv.gz", index=False,
                                                  compression="gzip")
        rec.to_csv(ddir / f"{split}_recovered.csv.gz", index=False, compression="gzip")
        summ.to_csv(ddir / f"{split}_versions.csv", index=False)
        lab = rec.label.value_counts().to_dict()
        print(f"  {len(summ)} versions, {int(summ.n_deleted.sum())} suspect records deleted, "
              f"{int(summ.n_foreign.sum())} foreign; recovered {len(rec)} {lab}  "
              f"[{time.time() - t0:.0f}s]")
    print("\nDone. Next: python n5_03_calibration.py")
