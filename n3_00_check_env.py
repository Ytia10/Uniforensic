"""N3 STEP 2 - Check the N3 libraries, N1's outputs, and the N3 building blocks.
Run:  python n3_00_check_env.py
"""
import importlib
import sys

EXPECT = {"apsw": "3.53.4.0", "sqlite_dissect": None, "sklearn": "1.9.1", "pandas": None, "matplotlib": None}
ok = True
for mod, want in EXPECT.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "") or (m.apswversion() if mod == "apsw" else "")
        flag = "OK" if (want is None or ver == want) else "WARN"
        print(f"  [{flag:4s}] {mod:15s} {ver}" + ("" if flag == "OK" else f"   (expected {want})"))
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  [FAIL] {mod:15s} {e}")
if not ok:
    sys.exit("Install the N3 libraries:  pip install -r requirements_n3.txt")

import apsw                                                  # noqa: E402

print(f"  [OK]   SQLite engine used to build databases: {apsw.sqlite_lib_version()} (bundled with APSW)")

from ufn3.common import load_config_n3, load_n1_cases       # noqa: E402
from ufn3.selftest import run_all                            # noqa: E402

cfg3 = load_config_n3()
for split in ("train", "calib", "test"):
    _, key, arts = load_n1_cases(split)
    print(f"  [OK]   N1 {split:5s} split: {len(key)} cases, "
          + ", ".join(f"{len(df)} {t}" for t, df in arts.items()))
for line in run_all():
    print(f"  [OK]   {line}")
print("\nAll N3 checks passed. Next: python n3_01_build_databases.py")
