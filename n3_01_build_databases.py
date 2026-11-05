"""N3 STEP 3 - Turn N1 cases into real SQLite databases, delete rows, keep the truth.

For each selected N1 case: an Android SMS database, an Android call-log database,
a Chrome history database and a payment-app database, filled with that case's
artifacts in time order.
The owner deletes planted evidence (70%) and some other rows; later activity
overwrites part of the freed space. Each database gets one SQLite condition
(secure_delete, auto_vacuum, WAL, page size) from config_n3.yaml.
Output: data_n3/dbs/<split>/<case>/<file>   and   data_n3/truth_<split>.json.gz
Run:  python n3_01_build_databases.py      (about 1 minute)
"""
import gzip
import json
import time

import numpy as np

from ufn1.common import banner, derive_seed
from ufn3.builder import build_database
from ufn3.common import N1_TYPES, load_config_n3, load_n1_cases, path_n3, pick_cases

cfg3 = load_config_n3()
root = path_n3(cfg3, "data")
conds = list(cfg3["conditions"])
t0 = time.time()
for split, n in cfg3["cases_per_split"].items():
    cfg1, key, arts = load_n1_cases(split)
    cases = pick_cases(key, n, cfg3["seed"], split)
    by_case = {t: {cid: g.to_dict("records") for cid, g in df.groupby("case_id")} for t, df in arts.items()}
    jobs = [(c, db) for c in cases for db in cfg3["databases"]]
    rng = np.random.default_rng(derive_seed(cfg3["seed"], "n3_conditions", split))
    assign = rng.permutation(np.resize(np.arange(len(conds)), len(jobs)))   # even spread
    truth = []
    for (case, db), ci in zip(jobs, assign):
        name = conds[ci]
        r = np.random.default_rng(derive_seed(cfg3["seed"], "n3_build", split, case["case_id"], db))
        gt = build_database(db, by_case[N1_TYPES[db]].get(case["case_id"], []), case, name, cfg3["conditions"][name],
                            cfg3, r, root / "dbs" / split / case["case_id"])
        gt["split"] = split
        truth.append(gt)
    with gzip.open(root / f"truth_{split}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(truth, f)
    vers = [v for g in truth for t in g["tables"].values() for v in t["versions"]]
    nd = sum(v["deleted"] for v in vers)
    ne = sum(v["deleted"] and v["role"] == "evidence" for v in vers)
    banner(f"{split}: {len(cases)} cases -> {len(truth)} databases, "
           f"{len(vers)} rows written, {nd} deleted ({ne} rows of planted evidence)")
    for name in conds:
        print(f"  {name:16s} {sum(g['condition'] == name for g in truth):4d} databases")
print(f"\nDone in {time.time() - t0:.0f}s. Next: python n3_02_carve.py")
