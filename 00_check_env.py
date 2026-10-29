"""STEP 2 - Check that every library imports and works together.
Run:  python 00_check_env.py
"""
import importlib
import platform
import sys

print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
if sys.version_info[:2] != (3, 12):
    print("WARNING: this project was verified on Python 3.12. Other versions may fail.")

ok = True
for mod in ["numpy", "pandas", "scipy", "sklearn", "torch", "captum", "shap",
            "xgboost", "matplotlib", "yaml"]:
    try:
        m = importlib.import_module(mod)
        print(f"  [OK]   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:                       # noqa: BLE001
        ok = False
        print(f"  [FAIL] {mod:12s} {e}")
if not ok:
    sys.exit("Fix the failed imports above (re-run: pip install -r requirements.txt).")

# --- smoke test: tiny model through every explainer + XGBoost TreeSHAP ---
import numpy as np
import torch

from ufn1.explainers import EXPLAINERS, run_explainer
from ufn1.metrics import self_test
from ufn1.models import TriageMLP, train_xgb, xgb_treeshap

self_test()
print("  [OK]   metric self-test (hand-computed answers match)")

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 6)).astype(np.float32)
y = np.digitize(X[:, 0], [-0.5, 0.5])          # 3 classes, like the real task
model = TriageMLP(6, 3, (8,), 0.0).eval()
cfg = {"kernelshap": {"nsamples": 64}, "integrated_gradients": {"n_steps": 8},
       "gradientshap": {"n_samples": 8}, "lime": {"n_samples": 50},
       "feature_ablation": {}, "random": {}}
for name in EXPLAINERS:
    a = run_explainer(name, model, X[:3], y[:3], X[:10], cfg[name], seed=0, progress=False)
    assert a.shape == (3, 6) and np.isfinite(a).all(), name
    print(f"  [OK]   explainer {name}")
clf = train_xgb(X, y, {"n_estimators": 5, "max_depth": 2}, seed=0)
ts = xgb_treeshap(clf, X[:3])
assert ts.shape[0] == 3 and ts.shape[-1] == 6
print("  [OK]   xgboost exact TreeSHAP")
print(f"\nAll checks passed. torch threads = {torch.get_num_threads()}")
