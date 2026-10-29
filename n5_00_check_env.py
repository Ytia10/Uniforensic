"""N5 STEP 2 - Check libraries, the outputs of N1/N4, the teammate's N3 files, and
the N5 building blocks (each against an independent reference).
Run:  python n5_00_check_env.py
"""
import hashlib
import importlib
import sys
import tempfile

ok = True
for mod in ["numpy", "pandas", "sklearn", "torch", "captum", "flwr", "case_utils", "mapie", "yaml"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  [FAIL] {mod:12s} {e}")
if not ok:
    sys.exit("Install the N5 libraries:  pip install -r requirements_n5.txt")

import sqlite3                                               # noqa: E402

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

print(f"  [OK]   SQLite library {sqlite3.sqlite_version} (writes the test databases; the guide's run used 3.45.1)")

from ufn1.common import ROOT, load_config                    # noqa: E402

# ---------------------------------------------------------------- inputs
cfg1 = load_config()
need = [ROOT / cfg1["paths"]["data"] / f"{s}_{t}.csv.gz" for s in ("train", "val", "test")
        for t in ("sms", "call", "web", "login", "txn")]
need += [ROOT / cfg1["paths"]["models"] / f for f in ("mlp.pt", "scaler.json")]
need += [ROOT / "data_n4" / "partitions.json", ROOT / "config_n2.yaml", ROOT / "config_n4.yaml"]
missing = [str(p.relative_to(ROOT)) for p in need if not p.exists()]
if missing:
    sys.exit("Missing inputs: " + ", ".join(missing) +
             "\nRun N1 (python run_all.py --skip-attack) and N4 (python n4_run_all.py) first.")
print("  [OK]   N1 benchmark + model, N4 partitions, N2/N4 configs found")

from ufn5.n3_adapter import N3_DIR, N3_FILES                  # noqa: E402
changed = []
for f, md5 in N3_FILES.items():
    p = N3_DIR / f
    if not p.exists():
        sys.exit(f"  [FAIL] teammate's N3 file missing: n3_team/{f}  (copy the 7 N3 .py files there)")
    if hashlib.md5(p.read_bytes()).hexdigest() != md5:
        changed.append(f)
if changed:
    print(f"  [WARN] n3_team files differ from the version N5 was verified with: {changed}\n"
          f"         N5 will run, but its numbers can differ from the guide.")
else:
    print("  [OK]   teammate's N3 files present and identical to the verified version (md5)")

# ------------------------------------------- 1. weighted features == N1 features
from ufn1.features import extract_features                   # noqa: E402
from ufn5.common import load_split                           # noqa: E402
from ufn5.wfeatures import extract_features_weighted         # noqa: E402
worst = 0.0
for split in ("train", "val", "test"):
    keys, tabs = load_split(split)
    ids = [k["case_id"] for k in keys]
    a = extract_features(tabs, ids)
    b = extract_features_weighted({t: d.assign(w=1.0) for t, d in tabs.items()}, ids)
    worst = max(worst, float((a - b).abs().max().max()))
assert worst < 1e-9, worst
print(f"  [OK]   weighted features with w = 1 equal N1's features on all 3,600 cases (max diff {worst:.1e})")

# ------------------------------------------- 2. artifact -> SQLite row -> artifact
from ufn5.devicedb import TABLES, from_row, to_row           # noqa: E402
cols = {"sms": ["timestamp", "direction", "counterparty", "text"],
        "call": ["timestamp", "direction", "counterparty", "answered", "duration_s"],
        "web": ["timestamp", "domain"], "login": ["timestamp", "service", "account", "ip_country", "success"],
        "txn": ["timestamp", "direction", "counterparty", "amount"]}
n = 0
for t in TABLES:
    for a in tabs[t].itertuples(index=False):
        back = from_row(t, to_row(t, a))
        for c in cols[t]:
            x, y = getattr(a, c), back[c]
            if c in ("answered", "success"):
                x = str(x).lower() == "true"
            if c == "counterparty":          # N1's CSV parses call numbers as integers
                x = str(x)
            assert x == y, (t, c, x, y)
        n += 1
print(f"  [OK]   {n} test artifacts survive artifact -> SQLite row -> artifact unchanged")

# ------------------------------------------- 3. one real build + carve
from ufn5.common import load_config_n5, per_case             # noqa: E402
from ufn5.n3_adapter import train_teammate_classifier        # noqa: E402
from ufn5.recover import run_case                            # noqa: E402
cfg5 = load_config_n5()
with tempfile.TemporaryDirectory() as tmp:
    from pathlib import Path
    clf, _ = train_teammate_classifier(Path(tmp), cfg5["n3_numpy_seed"])
k = next(k for k in keys if k["label"] == "fraud")
truth, rec = run_case(k["case_id"], per_case(tabs)[k["case_id"]], 1.0, False, clf,
                      cfg5["scenario"], cfg5["seed"], cfg1["benchmark"]["window_days"])
assert len(rec) > 0 and set(rec.label) <= {"correct", "foreign", "wrong"}
print(f"  [OK]   real SQLite build + N3 carving of one case: {int(truth.deleted.sum())} deleted, "
      f"{len(rec)} recovered ({(rec.label == 'correct').mean():.0%} correct)")

# ------------------------------------------- 4. conformal: own LAC == MAPIE, and coverage
from ufn5.conformal import coverage, lac_quantile, lac_sets, mapie_sets  # noqa: E402
rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(4) * 0.7, size=3000)
yv = np.array([rng.choice(4, p=p) for p in P])
cal, te = np.arange(1000), np.arange(1000, 3000)
q = lac_quantile(P[cal], yv[cal], 0.1)
S = lac_sets(P[te], q)
M = mapie_sets(P, cal, yv[cal], te, 0.1)
assert (S == M).all(), "own LAC sets differ from MAPIE"
cov = coverage(S, yv[te])
assert 0.87 < cov < 0.93, cov
print(f"  [OK]   own LAC prediction sets identical to MAPIE; coverage {cov:.3f} at target 0.90")
print("\nAll N5 checks passed.")
