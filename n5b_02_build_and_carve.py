"""N5b STEP 4 - Put every N1 case on a simulated phone, delete, carve with the ALTERNATIVE N3.

Identical to n5_02_build_and_carve.py (same versions, same deletion plans, same
databases byte for byte, same labels), except that each database is carved by the
alternative N3 (ufn3 carver + its calibrated ranker, see ufn5b/carver_ufn3.py).

Output (data_n5b/): {split}_versions.csv, {split}_deleted.csv.gz,
{split}_recovered.csv.gz, sample_dbs/.
Run:  python n5b_02_build_and_carve.py        (about 30-60 min on 2 CPU cores)
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed, load_config
from ufn5.common import SPLITS, load_split, path_n5, per_case, version_id
from ufn5b.carver_ufn3 import load_ranker
from ufn5b.common import load_config_n5b
from ufn5b.recover import run_case

cfg1, cfg5 = load_config(), load_config_n5b()
ddir, mdir = path_n5(cfg5, "data"), path_n5(cfg5, "models")
scfg, seed, levels = cfg5["scenario"], cfg5["seed"], cfg5["scenario"]["levels"]
window = cfg1["benchmark"]["window_days"]
RANKER = load_ranker(cfg5)
SAMPLE_DIR = ddir / "sample_dbs"
SAMPLE_DIR.mkdir(exist_ok=True)


def work(task):
    case_id, ctab, level, foreign, keep = task
    keep_dir = None
    if keep:
        keep_dir = SAMPLE_DIR / version_id(case_id, level, foreign).replace("|", "_")
        keep_dir.mkdir(exist_ok=True)
        keep_dir = str(keep_dir)
    truth, rec = run_case(case_id, ctab, level, foreign, RANKER, scfg, seed, window, keep_dir)
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
        banner(f"Split '{split}': build databases, delete, carve with the alternative N3")
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
    print("\nDone. Next: python n5b_03_calibration.py")
