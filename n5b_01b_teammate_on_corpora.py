"""N5b STEP 3b - The TEAMMATE's N3 on the same public corpora, scored the same way.

n3_07 tested the alternative N3 on two public corpora (Nemetz et al.; NIST CFReDS
SFT-03). The teammate's N3 was only tested on its own synthetic databases. This step
runs the teammate's code UNCHANGED (SQLiteCarver + the classifier n5_01 trained and
verified bit for bit, models_n5/n3_classifier.json) on the SAME files with the SAME
answer keys (ufn3.corpora), so both carvers get public numbers on identical data.

How the teammate's tool is used here (as N5 uses it, ufn5.n3_adapter.carve):
  * it is run once per table named in the answer key, with that table's columns and
    declared types (its schema-dict input); it reads the main database file only
    (it has no WAL / rollback-journal reader - this is part of what is measured);
  * a carved record is kept if its column count fits the table; exact copies of live
    rows are dropped (the more favourable setting: its own fuzzy live filter, which
    N5 showed also drops genuine deleted rows, is NOT applied);
  * the same content found twice counts once (highest p kept);
  * reported = p >= 0.5; rows NIST modified are neutral (as in n3_07).
Scoring (precision, recall of all / of still-present deleted rows, ECE with the same
15 bins, ROC AUC) uses the same functions as n3_07.

Output: results_n5b/teammate_n3_corpora_summary.csv  (same columns as corpora_summary.csv)
        results_n5b/carvers_on_public_corpora.csv    (side by side: teammate N3,
                                                       alternative N3, SQLite Dissect)
Run:  python n5b_01b_teammate_on_corpora.py      (about 1-5 minutes)
"""
import time

import numpy as np
import pandas as pd

from ufn1.common import ROOT, banner, load_json
from ufn3.common import load_config_n3, path_n3
from ufn3.corpora import SFT03_VARIANTS, cfreds_truth, nemetz_truth, verify
from ufn3.dissect import canon_value
from ufn3.nemetz import find_databases
from ufn3.sqlite_format import FormatError, open_image
from ufn3.truth import ece, ranking
from ufn5.common import path_n5
from ufn5.n3_adapter import SQLiteCarver, classifier_from_json
from ufn5b.common import load_config_n5b

cfg3, cfg5 = load_config_n3(), load_config_n5b()
rc = cfg3["ranker"]
third = path_n3(cfg3, "data") / "third_party"
rdir = path_n5(cfg5, "results")
if verify(third):
    raise SystemExit("Third-party corpora missing or changed - run the alternative N3 first "
                     "(python n3_06_get_corpora.py --source <folder>, then n3_07_corpora.py).")
clf_file = ROOT / "models_n5" / "n3_classifier.json"
if not clf_file.exists():
    raise SystemExit("models_n5/n3_classifier.json missing - run N5 step 1 first (python n5_01_n3_classifier.py).")
CLF = classifier_from_json(load_json(clf_file))
THR = 0.5

items = []
for i, p, x in find_databases(third / "nemetz", cfg3["nemetz"]["categories"] + cfg3["nemetz"]["robustness"]):
    items.append({"corpus": "nemetz", "group": i[:2], "id": i, "path": p, "xml": x, "wal": None, "journal": None})
for name, (side, kind) in SFT03_VARIANTS.items():
    items.append({"corpus": "nist_sft03", "group": name.split(".")[0], "id": name, "path": third / "cfreds" / name,
                  "wal": third / "cfreds" / side if kind == "wal" else None,
                  "journal": third / "cfreds" / side if kind == "journal" else None})

banner(f"Teammate's N3 (unchanged) on {len(items)} public databases")
recs, dbrows = [], []
t0 = time.time()
for it in items:
    db = it["path"].read_bytes()
    wal = it["wal"].read_bytes() if it["wal"] else None
    jrn = it["journal"].read_bytes() if it["journal"] else None
    blobs = [b for b in (db, wal, jrn) if b]
    try:
        img = open_image(db, wal, jrn)             # only to read the schema and build the answer key
    except FormatError as e:
        raise SystemExit(f"  [FAIL] could not read {it['id']}: {e}") from e
    truth = (nemetz_truth(img, it["xml"], blobs) if it["corpus"] == "nemetz"
             else cfreds_truth(third / "cfreds", it["id"], blobs))
    err = ""
    for table, t in truth.items():
        sch = img.tables[table]
        n = len(sch.columns)
        try:
            cands = SQLiteCarver({"table_name": table, "columns": sch.columns,
                                  "column_types": sch.types}).scan_database(str(it["path"]))
        except Exception as e:                      # noqa: BLE001  (a crash is a result too)
            err = f"{type(e).__name__}: {e}"
            cands = []
        if not cands:
            continue
        probs = CLF.predict_proba([c["features"] for c in cands])
        for c, p in zip(cands, probs):
            v = list(c["values"])
            if len(v) == n:
                k = tuple(canon_value(x) for j, x in enumerate(v) if j != sch.ipk)
            elif len(v) == n - 1 and sch.ipk is not None:
                k = tuple(canon_value(x) for x in v)          # rowid alias left out of the record
            else:
                continue                                      # cannot be a row of this table
            if k in t["live"]:
                continue                                      # exact live copy: dropped (as in N5)
            y = int(k in t["deleted"])
            recs.append({"corpus": it["corpus"], "id": it["id"], "table": table, "ckey": repr(k), "y": y,
                         "neutral": int(not y and k[:3] in t["neutral"]), "present": int(k in t["present"]),
                         "p": float(p)})
    dbrows.append({"corpus": it["corpus"], "group": it["group"], "id": it["id"],
                   "deleted": sum(len(t["deleted"]) for t in truth.values()),
                   "present": sum(len(t["present"]) for t in truth.values()), "error": err})
R = pd.DataFrame(recs, columns=["corpus", "id", "table", "ckey", "y", "neutral", "present", "p"])
C = R.groupby(["corpus", "id", "table", "ckey"], as_index=False, sort=False).agg(
    {"y": "max", "neutral": "max", "present": "max", "p": "max"})
DB = pd.DataFrame(dbrows)
print(f"  carved in {time.time() - t0:.0f}s; {len(C)} distinct non-live records; "
      f"crashed on {(DB.error != '').sum()} databases")


def summarise(corpus: str, groups: dict) -> list[dict]:
    out = []
    for gname, gids in groups.items():
        c = C[(C.corpus == corpus) & C.id.isin(gids) & (C.neutral == 0)]
        b = DB[(DB.corpus == corpus) & DB.id.isin(gids)]
        nd, npres = int(b.deleted.sum()), int(b.present.sum())
        rep = c[c.p >= THR]
        tp = int(rep.y.sum())
        out.append({"corpus": corpus, "group": gname, "method": "teammate_n3", "deleted": nd, "present": npres,
                    "reported": len(rep), "tp": tp, "fp": len(rep) - tp, "live_fp": 0,
                    "precision": tp / len(rep) if len(rep) else np.nan,
                    "recall_all": tp / nd if nd else np.nan,
                    "recall_present": int(rep.y[rep.present == 1].sum()) / npres if npres else np.nan,
                    "ece": ece(c.y, c.p, rc["ece_bins"]) if len(c) else np.nan,
                    "auc": ranking(c.y, c.p)[0] if len(c) else np.nan,
                    "crashed_on": int((b.error != "").sum())})
    return out


ids = DB.groupby(["corpus", "group"]).id.apply(list)
rec_cats = cfg3["nemetz"]["categories"]
groups_n = {"records (07+0C+0D+0E)": sum((ids["nemetz"][g] for g in rec_cats if g in ids["nemetz"]), [])}
groups_n.update({f"cat {g}": ids["nemetz"][g] for g in rec_cats + cfg3["nemetz"]["robustness"] if g in ids["nemetz"]})
groups_c = {"all 4 variants": sum(ids["nist_sft03"].tolist(), [])}
groups_c.update({g: v for g, v in ids["nist_sft03"].items()})
S = pd.DataFrame(summarise("nemetz", groups_n) + summarise("nist_sft03", groups_c))
S.to_csv(rdir / "teammate_n3_corpora_summary.csv", index=False)

# side by side with the alternative N3 and SQLite Dissect (n3_07's file, read-only)
alt = pd.read_csv(ROOT / "results_n3" / "corpora_summary.csv")
alt = alt[alt.method.isin(["ranker_cal", "sqlite_dissect"])]
both = pd.concat([S, alt], ignore_index=True)
for col in ("deleted", "present"):                          # the same answer keys must give the same counts
    chk = both.groupby(["corpus", "group"])[col].nunique()
    if (chk > 1).any():
        raise SystemExit(f"  [FAIL] '{col}' differs between this step and n3_07 for: {list(chk[chk > 1].index)}")
order = {"teammate_n3": 0, "ranker_cal": 1, "sqlite_dissect": 2}
both = both.assign(o=both.method.map(order)).sort_values(["corpus", "group", "o"]).drop(columns="o")
both.to_csv(rdir / "carvers_on_public_corpora.csv", index=False)
for corpus, main in (("nemetz", "records (07+0C+0D+0E)"), ("nist_sft03", "all 4 variants")):
    banner(f"{corpus}: {main}")
    s = both[(both.corpus == corpus) & (both.group == main)]
    print(f"  {int(s.deleted.iloc[0])} deleted rows, {int(s.present.iloc[0])} still present in the bytes")
    print(f"  {'method':16s} {'reported':>8s} {'TP':>5s} {'FP':>5s} {'precision':>9s} {'recall(all)':>11s} "
          f"{'recall(present)':>15s} {'ECE':>6s}")
    for _, r in s.iterrows():
        print(f"  {r.method:16s} {int(r.reported):8d} {int(r.tp):5d} {int(r.fp):5d} {r.precision:9.3f} "
              f"{r.recall_all:11.3f} {r.recall_present:15.3f} {r.ece:6.3f}")
banner("NIST SFT-03 per variant (the teammate's carver reads the main file only)")
s = both[(both.corpus == "nist_sft03") & (both.group != "all 4 variants")]
print(s[["group", "method", "reported", "tp", "recall_all"]].to_string(index=False))
banner("Nemetz anti-forensic categories 17 + 18 (nothing intact survives): false alarms")
for g in cfg3["nemetz"]["robustness"]:
    s = both[(both.corpus == "nemetz") & (both.group == f"cat {g}")].set_index("method")
    print(f"  cat {g}: reported by teammate N3 {int(s.loc['teammate_n3', 'reported'])}, alternative N3 "
          f"{int(s.loc['ranker_cal', 'reported'])}, sqlite_dissect {int(s.loc['sqlite_dissect', 'reported'])}")
print("\nDone. Next: python n5b_02_build_and_carve.py")
