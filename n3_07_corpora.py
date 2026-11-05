"""N3 STEP 9 - Third-party test: databases N3 never trained on, built by others.

  Nemetz corpus, record categories 07 / 0C / 0D / 0E (deleted, overwritten and
  overflowing records) - scored on recall and precision;
  Nemetz anti-forensic categories 17 / 18 (manipulated freeblocks / freelist) -
  robustness: no row survives intact, so every reported record is a false alarm;
  NIST CFReDS SFT-03 (Android + iOS, WAL + PERSIST journal) - 100 deleted rows each.
Methods: strict carving, fixed heuristic, the calibrated ranker of step 5, two
ablations (ranker + content feature; ranker trained without the payment-app
tables), and DC3 SQLite Dissect as an external tool. SQLite Dissect is also run
on a stratified sample of the synthetic TEST databases, for comparison there.
Output: results_n3/corpora_summary.csv, results_n3/corpora_records.csv.gz,
        results_n3/dissect_synthetic.csv
Run:  python n3_07_corpora.py      (about 10-15 minutes; SQLite Dissect is slow on some files)
"""
import gzip
import json
import time

import joblib
import numpy as np
import pandas as pd

from ufn1.common import banner, load_json
from ufn3.carver import CONTENT_FEATURES, FEATURES, MODEL_FEATURES, carve, group_candidates, heuristic_score, is_strict
from ufn3.common import load_config_n3, path_n3
from ufn3.corpora import SFT03_VARIANTS, cfreds_truth, nemetz_truth, verify
from ufn3.dissect import canon_text, canon_value, run
from ufn3.nemetz import find_databases
from ufn3.ranker import calibrate, candidates, fit_ranker, matrix, read_records
from ufn3.sqlite_format import open_image
from ufn3.truth import ece, ranking

cfg3 = load_config_n3()
rc = cfg3["ranker"]
data, models, results = (path_n3(cfg3, k) for k in ("data", "models", "results"))
third = data / "third_party"
if verify(third):
    raise SystemExit("Third-party corpora missing or changed - run step 8 first (python n3_06_get_corpora.py).")
bundle = joblib.load(models / "ranker.joblib")
t95 = load_json(models / "ranker.json")["t95"]

banner("Training the two ablation rankers (same data and settings as step 5)")
tr, ca = (candidates(read_records(data / f"records_{s}.csv.gz")) for s in ("train", "calib"))
with_content = MODEL_FEATURES + CONTENT_FEATURES
abl = {"abl_content": (calibrate(fit_ranker(tr, rc, cfg3["seed"], with_content), ca, rc["calibration"], with_content),
                       with_content),
       "abl_no_pay": (calibrate(fit_ranker(tr[tr.db != "pay"], rc, cfg3["seed"]), ca[ca.db != "pay"],
                                rc["calibration"]), MODEL_FEATURES)}
print("  done")

items = []
for i, p, x in find_databases(third / "nemetz", cfg3["nemetz"]["categories"] + cfg3["nemetz"]["robustness"]):
    items.append({"corpus": "nemetz", "group": i[:2], "id": i, "path": p, "xml": x, "wal": None, "journal": None})
for name, (side, kind) in SFT03_VARIANTS.items():
    items.append({"corpus": "nist_sft03", "group": name.split(".")[0], "id": name, "path": third / "cfreds" / name,
                  "wal": third / "cfreds" / side if kind == "wal" else None,
                  "journal": third / "cfreds" / side if kind == "journal" else None})

recs, dis, dbrows = [], [], []
t0 = time.time()
for it in items:
    db = it["path"].read_bytes()
    wal = it["wal"].read_bytes() if it["wal"] else None
    jrn = it["journal"].read_bytes() if it["journal"] else None
    blobs = [b for b in (db, wal, jrn) if b]
    img = open_image(db, wal, jrn)
    cands, info = carve(img, cfg3["carver"]["max_candidates_per_region"])
    grouped = group_candidates(img, cands, info["bounds"])
    truth = (nemetz_truth(img, it["xml"], blobs) if it["corpus"] == "nemetz"
             else cfreds_truth(third / "cfreds", it["id"], blobs))
    for r in grouped:
        t = truth.get(r.table)
        sch = img.tables[r.table]
        k = tuple(canon_value(v) for j, v in enumerate(r.values) if j != sch.ipk)
        y = int(t is not None and k in t["deleted"])
        neutral = int(t is not None and not y and k[:3] in t["neutral"])
        recs.append({"corpus": it["corpus"], "group": it["group"], "id": it["id"], "table": r.table,
                     "ckey": repr(k), "in_truth_table": int(t is not None), "y": y, "neutral": neutral,
                     "live_dup": int(r.live_duplicate), "present": int(t is not None and k in t["present"]),
                     "strict": int(is_strict(r.features)), "heuristic": heuristic_score(r.features),
                     "best_hypothesis": r.best.hypothesis, **{f: r.features[f] for f in FEATURES}})
    rows, secs, err = run(it["path"])
    found = set()
    for table, vals in rows:
        t = truth.get(table)
        sch = img.tables.get(table)
        if sch is None:                                   # an index or other non-table output
            continue
        k = tuple(canon_text(v) for j, v in enumerate(vals) if j != sch.ipk)
        if (table, k) in found:
            continue                                      # the same row reported twice counts once
        found.add((table, k))
        kind = ("tp" if t and k in t["deleted"] else "neutral" if t and k[:3] in t["neutral"]
                else "live_fp" if t and k in t["live"] else "fp")
        dis.append({"corpus": it["corpus"], "group": it["group"], "id": it["id"], "kind": kind})
    dbrows.append({"corpus": it["corpus"], "group": it["group"], "id": it["id"],
                   "deleted": sum(len(t["deleted"]) for t in truth.values()),
                   "present": sum(len(t["present"]) for t in truth.values()), "dissect_error": err,
                   "dissect_seconds": round(secs, 2)})
R = pd.DataFrame(recs)
R["p"] = bundle["calibrators"][rc["calibration"]].predict_proba(matrix(R, bundle["features"]))[:, 1]
for name, (cal, feats) in abl.items():
    R[name] = cal.predict_proba(matrix(R, feats))[:, 1]
R["strict_score"] = R["strict"].astype(float)
# Two recovered records can differ in stored form (e.g. 1 vs 1.0) yet show the same values;
# the answer keys compare shown values, so such records are merged and keep their best score.
keys = ["corpus", "group", "id", "table", "ckey"]
agg = {c: "max" for c in ["y", "neutral", "present", "live_dup", "in_truth_table", "strict", "strict_score",
                          "heuristic", "p", *abl]}
agg["best_hypothesis"] = "first"
C = candidates(R.groupby(keys, as_index=False, sort=False).agg(agg))
D, DB = pd.DataFrame(dis), pd.DataFrame(dbrows)
print(f"  carved {len(items)} databases in {time.time() - t0:.0f}s")

METHODS = [("strict", "strict_score", 1.0), ("heuristic", "heuristic", 0.5), ("ranker_cal", "p", rc["report_threshold"]),
           ("ranker_cal_t95", "p", t95), ("abl_content", "abl_content", 0.5), ("abl_no_pay", "abl_no_pay", 0.5)]


def summarise(corpus: str, groups: dict) -> list[dict]:
    out = []
    for gname, gids in groups.items():
        c = C[(C.corpus == corpus) & C.id.isin(gids) & (C.neutral == 0)]
        b = DB[(DB.corpus == corpus) & DB.id.isin(gids)]
        nd, npres = int(b.deleted.sum()), int(b.present.sum())
        for m, col, thr in METHODS:
            rep = c[c[col] >= thr]
            tp = int(rep.y.sum())
            out.append({"corpus": corpus, "group": gname, "method": m, "deleted": nd, "present": npres,
                        "reported": len(rep), "tp": tp, "fp": len(rep) - tp, "live_fp": 0,
                        "precision": tp / len(rep) if len(rep) else np.nan,
                        "recall_all": tp / nd if nd else np.nan,
                        "recall_present": int(rep.y[rep.present == 1].sum()) / npres if npres else np.nan,
                        "ece": ece(c.y, c[col], rc["ece_bins"]) if m != "strict" and len(c) else np.nan,
                        "auc": ranking(c.y, c[col])[0] if m != "strict" and len(c) else np.nan})
        d = D[(D.corpus == corpus) & D.id.isin(gids)]
        tp, fp, lfp = int((d.kind == "tp").sum()), int((d.kind == "fp").sum()), int((d.kind == "live_fp").sum())
        out.append({"corpus": corpus, "group": gname, "method": "sqlite_dissect", "deleted": nd, "present": npres,
                    "reported": tp + fp + lfp, "tp": tp, "fp": fp + lfp, "live_fp": lfp,
                    "precision": tp / (tp + fp + lfp) if tp + fp + lfp else np.nan,
                    "recall_all": tp / nd if nd else np.nan, "recall_present": np.nan, "ece": np.nan,
                    "auc": np.nan, "crashed_on": int((b.dissect_error != "").sum())})
    return out


ids = DB.groupby(["corpus", "group"]).id.apply(list)
rec_cats = cfg3["nemetz"]["categories"]
groups_n = {"records (07+0C+0D+0E)": sum((ids["nemetz"][g] for g in rec_cats if g in ids["nemetz"]), [])}
groups_n.update({f"cat {g}": ids["nemetz"][g] for g in rec_cats + cfg3["nemetz"]["robustness"] if g in ids["nemetz"]})
groups_c = {"all 4 variants": sum(ids["nist_sft03"].tolist(), [])}
groups_c.update({g: v for g, v in ids["nist_sft03"].items()})
S = pd.DataFrame(summarise("nemetz", groups_n) + summarise("nist_sft03", groups_c))
S.to_csv(results / "corpora_summary.csv", index=False)
C.to_csv(results / "corpora_records.csv.gz", index=False)
DB.to_csv(results / "corpora_databases.csv", index=False)

for corpus, main in (("nemetz", "records (07+0C+0D+0E)"), ("nist_sft03", "all 4 variants")):
    banner(f"{corpus}: {main}")
    s = S[(S.corpus == corpus) & (S.group == main)]
    r0 = s.iloc[0]
    print(f"  {int(r0.deleted)} deleted rows, {int(r0.present)} still present in the bytes")
    print(f"  {'method':16s} {'reported':>8s} {'TP':>5s} {'FP':>5s} {'precision':>9s} {'recall(all)':>11s} "
          f"{'recall(present)':>15s} {'ECE':>6s}")
    for _, r in s.iterrows():
        print(f"  {r.method:16s} {r.reported:8d} {r.tp:5d} {r.fp:5d} {r.precision:9.3f} {r.recall_all:11.3f} "
              f"{r.recall_present:15.3f} {r.ece:6.3f}" + (f"   (crashed on {int(r.crashed_on)} of "
                                                           f"{len(groups_n[main] if corpus == 'nemetz' else groups_c[main])} "
                                                           "databases)" if r.method == "sqlite_dissect" and r.crashed_on else ""))
banner("Nemetz anti-forensic categories 17 + 18 (nothing intact survives): false alarms")
for g in cfg3["nemetz"]["robustness"]:
    s = S[(S.corpus == "nemetz") & (S.group == f"cat {g}")].set_index("method")
    print(f"  cat {g}: reported by strict {s.loc['strict', 'reported']}, ranker {s.loc['ranker_cal', 'reported']}, "
          f"sqlite_dissect {s.loc['sqlite_dissect', 'reported']} (crashed on {int(s.loc['sqlite_dissect', 'crashed_on'])})")
# ---------------- SQLite Dissect on synthetic test databases (stratified sample)
per = cfg3["dissect_sample_per_condition"]
with gzip.open(data / "truth_test.json.gz", "rt", encoding="utf-8") as f:
    truth_test = json.load(f)
by_cond: dict = {}
for gt in truth_test:
    by_cond.setdefault(gt["condition"], []).append(gt)
rng = np.random.default_rng(cfg3["seed"])
rows = []
for cond, gts in sorted(by_cond.items()):
    for k in sorted(rng.choice(len(gts), size=min(per, len(gts)), replace=False)):
        gt = gts[k]
        path = data / "dbs" / "test" / gt["case_id"] / gt["file"]
        dk = {(t, tuple(canon_value(v) for j, v in enumerate(v_["values"]) if j != T["ipk"])): v_["deleted"]
              for t, T in gt["tables"].items() for v_ in T["versions"]}
        ipk = {t: T["ipk"] for t, T in gt["tables"].items()}
        out, secs, err = run(path)
        seen = set()
        tp = fp = lfp = 0
        for table, vals in out:
            if table not in ipk:
                continue
            k2 = (table, tuple(canon_text(v) for j, v in enumerate(vals) if j != ipk[table]))
            if k2 in seen:
                continue
            seen.add(k2)
            if dk.get(k2) is True:
                tp += 1
            elif dk.get(k2) is False:
                lfp += 1
            else:
                fp += 1
        rows.append({"case_id": gt["case_id"], "db": gt["db"], "condition": cond, "tp": tp, "fp": fp,
                     "live_fp": lfp, "deleted": sum(v is True for v in dk.values()), "error": err,
                     "seconds": round(secs, 2)})
SD = pd.DataFrame(rows)
SD.to_csv(results / "dissect_synthetic.csv", index=False)
banner(f"SQLite Dissect on {len(SD)} synthetic test databases ({per} per condition)")
te = pd.read_csv(results / "test_scores.csv.gz", usecols=["case_id", "db", "y", "p"])
st = pd.read_csv(data / "dbstats_test.csv")
pick = SD[["case_id", "db"]]
ours = te.merge(pick, on=["case_id", "db"])
ours = ours[ours.p >= rc["report_threshold"]]
pres = st.merge(pick, on=["case_id", "db"])
rep = SD.tp.sum() + SD.fp.sum() + SD.live_fp.sum()
print(f"  same databases: {int(pres.deleted.sum())} deleted rows, {int(pres.present.sum())} present")
print(f"  sqlite_dissect: reported {rep}, correct {SD.tp.sum()}, live rows reported as deleted {SD.live_fp.sum()}, "
      f"other wrong {SD.fp.sum()}  -> precision {SD.tp.sum() / max(1, rep):.3f}, recall(present) "
      f"{SD.tp.sum() / max(1, pres.present.sum()):.3f}; crashed on {(SD.error != '').sum()} databases")
print(f"  ranker_cal:     reported {len(ours)}, correct {int(ours.y.sum())}, live rows reported 0 -> precision "
      f"{ours.y.mean():.3f}, recall(present) {ours.y.sum() / max(1, pres.present.sum()):.3f}")
print("\nDone. Next: python n3_08_export_for_n5.py")
