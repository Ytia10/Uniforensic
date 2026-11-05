"""N3 STEP 4 - Carve every database and label each recovered record with the truth.

The carver reads only the raw bytes (main file + WAL). After carving, each
recovered record is compared with the truth log: correct (equals a deleted row),
live duplicate (equals a live row - never reported) or wrong.
Output: data_n3/records_<split>.csv.gz   one row per recovered record + features
        data_n3/deleted_<split>.csv.gz   one row per deleted row (was it present?)
        data_n3/dbstats_<split>.csv      one row per database
Run:  python n3_02_carve.py      (about 2-4 minutes)
"""
import gzip
import json
import time

import pandas as pd

from ufn1.common import banner
from ufn3.carver import FEATURES, carve, group_candidates, heuristic_score, is_strict
from ufn3.common import load_config_n3, path_n3
from ufn3.sqlite_format import open_image
from ufn3.truth import truth_sets

cfg3 = load_config_n3()
root = path_n3(cfg3, "data")


def _json_value(v):
    return v.hex() if isinstance(v, (bytes, bytearray)) else v


t_all = time.time()
for split in cfg3["cases_per_split"]:
    with gzip.open(root / f"truth_{split}.json.gz", "rt", encoding="utf-8") as f:
        truth = json.load(f)
    recs, dels, stats = [], [], []
    t0 = time.time()
    for gt in truth:
        path = root / "dbs" / split / gt["case_id"] / gt["file"]
        side = {k: path.with_name(path.name + k) for k in ("-wal", "-journal")}
        db = path.read_bytes()
        wal, jrn = (side[k].read_bytes() if side[k].exists() else None for k in ("-wal", "-journal"))
        t1 = time.time()
        img = open_image(db, wal, jrn)
        cands, info = carve(img, cfg3["carver"]["max_candidates_per_region"])
        grouped = group_candidates(img, cands, info["bounds"])
        secs = time.time() - t1
        ts = truth_sets(gt, [b for b in (db, wal, jrn) if b])
        base = {"split": split, "case_id": gt["case_id"], "label": gt["label"], "tier": gt["tier"],
                "db": gt["db"], "condition": gt["condition"]}
        for r in grouped:
            k = (r.table, r.key)
            meta = ts["meta"].get(k, {})
            row = dict(base, table=r.table, rowid=r.rowid,
                       values=json.dumps([_json_value(v) for v in r.values]),
                       y=int(k in ts["deleted"]), live_dup=int(r.live_duplicate),
                       present=int(k in ts["present"]), artifact_id=meta.get("artifact_id", ""),
                       role=meta.get("role", ""), pattern_id=meta.get("pattern_id", ""),
                       heuristic=heuristic_score(r.features), strict=int(is_strict(r.features)),
                       best_hypothesis=r.best.hypothesis, best_source=r.best.source,
                       best_region=r.best.region)
            row.update({k: r.features[k] for k in FEATURES})
            recs.append(row)
        for k in sorted(ts["deleted"], key=str):
            v = ts["meta"][k]
            dels.append(dict(base, table=k[0], artifact_id=v["artifact_id"], role=v["role"],
                             pattern_id=v["pattern_id"], present=int(k in ts["present"])))
        stats.append(dict(base, rows=sum(len(t["versions"]) for t in gt["tables"].values()),
                          deleted=len(ts["deleted"]),
                          present=len(ts["present"]), candidates=info["candidates"],
                          records=len(grouped), live_dup=sum(r.live_duplicate for r in grouped),
                          overflow_missing=info["overflow_missing"], page_copies=info["page_copies"],
                          wal_frames=img.wal_info.get("frames", 0), seconds=round(secs, 4)))
    pd.DataFrame(recs).to_csv(root / f"records_{split}.csv.gz", index=False)
    pd.DataFrame(dels).to_csv(root / f"deleted_{split}.csv.gz", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(root / f"dbstats_{split}.csv", index=False)
    R = pd.DataFrame(recs)
    nl = R[R.live_dup == 0]
    banner(f"{split}: {len(truth)} databases carved in {time.time() - t0:.0f}s")
    print(f"  deleted rows: {st.deleted.sum()}   still present in the bytes: {st.present.sum()} "
          f"({st.present.sum() / st.deleted.sum():.1%})")
    print(f"  recovered records: {len(R)}  = {int(R.live_dup.sum())} live duplicates (never reported) + "
          f"{len(nl)} candidates ({int(nl.y.sum())} correct, {int((1 - nl.y).sum())} wrong)")
    print(f"  correct records found / present deleted rows: {nl.y.sum() / st.present.sum():.1%}  "
          "(the most any ranking of these candidates can reach)")
print(f"\nDone in {time.time() - t_all:.0f}s. Next: python n3_03_train_ranker.py")
