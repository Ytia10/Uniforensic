"""N2 STEP 2 - Check N1 and N4 outputs exist and the N2 building blocks are correct.
Run:  python n2_00_check_env.py
"""
import numpy as np

from ufn2.attacks import self_test as attacks_test
from ufn2.common import load_config_n2, load_n1, load_n4
from ufn2.dp import self_test as dp_test
from ufn2.summaries import block_sums, clip_rows, evidence_recall_at_3, release

cfg2 = load_config_n2()
cfg1, data, keys = load_n1()
print(f"  [OK]   N1 outputs found ({len(data['train']['y'])} train / {len(data['test']['y'])} test cases)")
cfg4, parts, mdir = load_n4()
d = cfg2["deployed"]
missing = [s for s in cfg4["run_seeds"] if not (mdir / f"fed_{d['method']}_a{d['alpha']}_s{s}.pt").exists()]
if missing:
    raise SystemExit(f"N4 federated models missing for seeds {missing}. Run N4 first.")
print(f"  [OK]   N4 outputs found (deployed model: {d['method']}, alpha {d['alpha']}, seeds {cfg4['run_seeds']})")

dp_test()
print("  [OK]   DP noise calibration (analytic Gaussian, incl. Monte-Carlo check of the definition)")
attacks_test()
print("  [OK]   attacks: leak found where certain, none found where impossible")

# summaries: clipping bound respected; secure aggregation = sum of agencies
rng = np.random.default_rng(0)
v = clip_rows(rng.normal(size=(50, 27)) * 5, 1.0)
assert np.all(np.linalg.norm(v, axis=1) <= 1.0 + 1e-9)
cls = rng.integers(0, 4, 50)
blocks = [block_sums(v[i::2], cls[i::2], 4) for i in range(2)]
whole = block_sums(v, cls, 4)
g = release("secagg", blocks, 1.0, float("inf"), 1e-5, rng)
assert np.allclose(g, whole[0] / np.maximum(whole[1], 1)[:, None])
assert evidence_recall_at_3(np.array([0.1, 5, 4, 3, 0.2]), [1, 2, 3]) == 1.0
print("  [OK]   summaries: clipping, secure-aggregation sum, evidence recall")
print("\nAll N2 checks passed.")
