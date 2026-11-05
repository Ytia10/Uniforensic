"""N5b STEP 2 - Check that everything N5b needs is in place, before anything is built.

N5b = N5 re-run with the alternative N3 carver (ufn3). It needs:
  * N1 outputs (data/, models/), N4's partition file, N2/N4 configs  (as N5)
  * the alternative N3 RUN, including its public-corpus test:
      models_n3/ranker.joblib + ranker.json   (n3_03_train_ranker.py)
      results_n3/corpora_summary.csv          (n3_07_corpora.py: Nemetz + NIST SFT-03)
  * the same scikit-learn version the ranker was saved with (joblib file)
Then it builds and carves ONE real case with N5's own code + the ufn3 carver.
Run:  python n5b_00_check_env.py
"""
import importlib
import sys

ok = True
for mod in ["numpy", "pandas", "sklearn", "joblib", "torch", "captum", "flwr", "case_utils", "mapie", "yaml"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  [FAIL] {mod:12s} {e}")
if not ok:
    sys.exit("Install the libraries:  pip install -r requirements_n5.txt -r requirements_n3.txt")

import sqlite3                                               # noqa: E402

import sklearn                                               # noqa: E402

from ufn1.common import ROOT, load_config, load_json         # noqa: E402

print(f"  [OK]   SQLite library {sqlite3.sqlite_version} (writes the device databases, as in N5)")

cfg1 = load_config()
need = [ROOT / cfg1["paths"]["data"] / f"{s}_{t}.csv.gz" for s in ("train", "val", "test")
        for t in ("sms", "call", "web", "login", "txn")]
need += [ROOT / cfg1["paths"]["models"] / f for f in ("mlp.pt", "scaler.json")]
need += [ROOT / "data_n4" / "partitions.json", ROOT / "config_n2.yaml", ROOT / "config_n4.yaml",
         ROOT / "models_n5" / "n3_classifier.json", ROOT / "n3_team" / "carver.py",
         ROOT / "config_n5b.yaml", ROOT / "ufn3" / "carver.py"]
missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
if missing:
    sys.exit("Missing inputs: " + ", ".join(missing) +
             "\nN5b needs N1 (python run_all.py --skip-attack), N4 (python n4_run_all.py), N5 step 1 "
             "(python n5_01_n3_classifier.py: the teammate's classifier, for step 3b) and the "
             "N3 add-on files (ufn3/, n3_*.py, config_n3.yaml) in this folder.")
print("  [OK]   N1 benchmark + model, N4 partitions, N2/N4 configs, teammate classifier, ufn3 package found")

from ufn5b.common import load_config_n5b                     # noqa: E402

cfg5 = load_config_n5b()
rk = ROOT / cfg5["carver"]["ranker"]
corp = ROOT / "results_n3" / "corpora_summary.csv"
if not rk.exists() or not rk.with_suffix(".json").exists():
    sys.exit(f"  [FAIL] {rk.relative_to(ROOT)} missing: run the alternative N3 first "
             "(python n3_run_all.py --source ext_sqlite_forensic).")
if not corp.exists():
    sys.exit("  [FAIL] results_n3/corpora_summary.csv missing: the alternative N3 has not been tested on the "
             "public corpora yet. Run n3_06_get_corpora.py and n3_07_corpora.py (README_N3.md, Step 8-9).")
saved = load_json(rk.with_suffix(".json"))["sklearn"]
if saved != sklearn.__version__:
    sys.exit(f"  [FAIL] the ranker was saved with scikit-learn {saved}, this environment has "
             f"{sklearn.__version__}: pip install scikit-learn=={saved}")
print(f"  [OK]   alternative N3 ranker found (scikit-learn {saved}) and public-corpus results present")

# ------------------------------------------- one real build + carve with the ufn3 carver
from ufn5.common import load_split, per_case                 # noqa: E402
from ufn5b.carver_ufn3 import load_ranker                    # noqa: E402
from ufn5b.recover import run_case                           # noqa: E402

keys, tabs = load_split("test")
ranker = load_ranker(cfg5)
k = next(k for k in keys if k["label"] == "fraud")
truth, rec = run_case(k["case_id"], per_case(tabs)[k["case_id"]], 1.0, False, ranker,
                      cfg5["scenario"], cfg5["seed"], cfg1["benchmark"]["window_days"])
assert len(rec) > 0 and set(rec.label) <= {"correct", "foreign", "wrong"}
assert rec.p_raw.between(0, 1).all()
print(f"  [OK]   real SQLite build + alternative-N3 carving of one case: {int(truth.deleted.sum())} deleted, "
      f"{len(rec)} recovered ({(rec.label == 'correct').mean():.0%} correct)")
print("\nAll N5b checks passed. Next: python n5b_01_carver.py")
