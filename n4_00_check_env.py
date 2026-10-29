"""N4 STEP 2 - Check the extra libraries, N1's outputs, and the N4 building blocks.
Run:  python n4_00_check_env.py
"""
import importlib
import sys

ok = True
for mod in ["flwr", "case_utils", "rdflib", "pyshacl", "torch", "captum", "shap"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:                                   # noqa: BLE001
        ok = False
        print(f"  [FAIL] {mod:12s} {e}")
if not ok:
    sys.exit("Install the N4 libraries:  pip install -r requirements_n4.txt")

import numpy as np                                           # noqa: E402

from ufn1.schema import CLASSES                              # noqa: E402
from ufn4.common import load_n1                              # noqa: E402
from ufn4.fl import combine, get_params, set_params          # noqa: E402
from ufn4.partition import class_counts, dirichlet_partition  # noqa: E402

cfg1, data, keys = load_n1()
print(f"  [OK]   N1 outputs found: {len(data['train']['y'])} train / {len(data['test']['y'])} test cases")

# Dirichlet split: every case used exactly once, no agency below the minimum
y = data["train"]["y"]
parts = dirichlet_partition(y, 8, 0.5, seed=0, min_size=30)
allidx = np.concatenate(parts)
assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)
assert class_counts(parts, y, len(CLASSES)).sum() == len(y)
print("  [OK]   Dirichlet split uses every case exactly once")

# Flower aggregation == hand-computed weighted mean / median
a = [np.array([1.0, 2.0]), np.array([3.0])]
b = [np.array([3.0, 6.0]), np.array([5.0])]
avg = combine([(a, 1), (b, 3)], "fedavg")
assert np.allclose(avg[0], [2.5, 5.0]) and np.allclose(avg[1], [4.5])
med = combine([(a, 1), (b, 3), (a, 1)], "fedmedian")
assert np.allclose(med[0], [1.0, 2.0])
print("  [OK]   Flower FedAvg / FedMedian match hand-computed values")

from ufn4.common import new_mlp                              # noqa: E402
m1, m2 = new_mlp(cfg1), new_mlp(cfg1)
set_params(m2, get_params(m1))
assert all(np.array_equal(p, q) for p, q in zip(get_params(m1), get_params(m2)))
print("  [OK]   model parameters copy exactly between server and agencies")
print("\nAll N4 checks passed.")
