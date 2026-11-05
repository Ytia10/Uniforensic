"""N5b STEP 3 - Record WHICH carver N5b uses, and that it was tested on public data.

N5 trains the teammate's classifier here (n5_01). N5b trains nothing: it uses the
alternative N3's ranker exactly as n3_03_train_ranker.py saved it. This step records
its identity (sha256 of the ranker file, md5 of every ufn3 file) and copies the
carver's public-corpus results (n3_07_corpora.py) next to the N5b results, so every
N5b number can be traced to a carver that was tested on the Nemetz corpus and NIST
CFReDS SFT-03.

Output: results_n5b/carver_ufn3.json
Run:  python n5b_01_carver.py
"""
import pandas as pd

from ufn1.common import ROOT, banner, load_json, save_json, sha256_file
from ufn5.common import path_n5
from ufn5b.common import load_config_n5b, ufn3_md5

cfg5 = load_config_n5b()
rdir = path_n5(cfg5, "results")
rk = ROOT / cfg5["carver"]["ranker"]
meta = load_json(rk.with_suffix(".json"))

banner("The carver N5b uses: the alternative N3 (ufn3)")
print(f"  ranker: {rk.relative_to(ROOT)}  sha256 {sha256_file(rk)[:16]}...")
print(f"  calibration: {meta['calibration']} (fit on {meta['calib_candidates']} candidates of N1 validation cases)")
print(f"  report threshold p >= {meta['report_threshold']}  (the same threshold N5 uses for variant A)")

cs = pd.read_csv(ROOT / "results_n3" / "corpora_summary.csv")
pub = cs[cs.method == "ranker_cal"].copy()
show = pub[pub.group.str.contains("records|all", case=False)][
    ["corpus", "group", "deleted", "present", "reported", "precision", "recall_all", "recall_present", "ece"]]
banner("Its test on public corpora (results_n3/corpora_summary.csv, never used for training)")
print(show.to_string(index=False))

save_json({"carver": "ufn3 (alternative N3)", "ranker_file": str(rk.relative_to(ROOT)),
           "ranker_sha256": sha256_file(rk), "ranker_meta": meta, "ufn3_md5": ufn3_md5(),
           "public_corpus_results": show.to_dict(orient="records")},
          rdir / "carver_ufn3.json")
print("\nDone. Next: python n5b_01b_teammate_on_corpora.py")
