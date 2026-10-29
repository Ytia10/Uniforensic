"""STEP 5 - Sanity-check the explainer + metric code on data with a KNOWN answer.

Before trusting any number on forensic cases, check that the same code finds the
right features where the answer is mathematically known (the idea behind
OpenXAI's synthetic generator, Agarwal et al., NeurIPS 2022):

  x ~ N(0, I) with 10 independent features
  y = 1  if  w.x + noise > 0,   w = [2.0, -1.5, 1.2, 1.0, 0, 0, 0, 0, 0, 0]

Local ground truth for case x: the 3 features with largest |w_i * x_i|
(their contribution to the logit). A correct explainer must find them far more
often than random. If this step fails, do NOT go further - the bug is in the code,
not in your forensic data.
Run:  python 03_sanity_check.py
"""
import numpy as np

from ufn1.common import banner, get_path, load_config, save_json, set_all_seeds
from ufn1.explainers import EXPLAINERS, run_explainer
from ufn1.metrics import bootstrap_ci, recall_at_k, self_test
from ufn1.models import mlp_predict_proba, train_mlp

cfg = load_config()
seed = cfg["seed"]
set_all_seeds(seed)
res_dir = get_path(cfg, "results")

banner("5a. Metric self-test (hand-computed answers)")
self_test()
print("  passed")

banner("5b. Known-answer synthetic data")
rng = np.random.default_rng(seed)
d, n = 10, 4000
w = np.array([2.0, -1.5, 1.2, 1.0, 0, 0, 0, 0, 0, 0])
X = rng.normal(size=(n, d)).astype(np.float32)
y = ((X @ w + rng.normal(scale=0.5, size=n)) > 0).astype(int)
Xtr, ytr, Xva, yva, Xte, yte = X[:3000], y[:3000], X[3000:3500], y[3000:3500], X[3500:], y[3500:]
model = train_mlp(Xtr, ytr, Xva, yva, cfg["mlp"], n_classes=2, seed=seed, verbose=False)
acc = (mlp_predict_proba(model, Xte).argmax(1) == yte).mean()
print(f"  synthetic MLP test accuracy = {acc:.3f}")

# explain the predicted class of 200 correctly classified test cases
pred = mlp_predict_proba(model, Xte).argmax(1)
idx = np.where(pred == yte)[0][:200]
Xe, ye = Xte[idx], yte[idx]
truth = [set(np.argsort(-np.abs(w * x))[:3]) for x in Xe]
bg = Xtr[rng.choice(len(Xtr), 50, replace=False)]

results = {}
for name in EXPLAINERS:
    attr = run_explainer(name, model, Xe, ye, bg, cfg["explainers"][name], seed, progress=False)
    m, lo, hi = bootstrap_ci([recall_at_k(a, g, 3) for a, g in zip(attr, truth)], 1000, seed)
    results[name] = {"recall@3": m, "ci95": [lo, hi]}
    print(f"  {name:22s} recall@3 = {m:.3f}  [{lo:.3f}, {hi:.3f}]")

rand = results["random"]["recall@3"]
failed = [k for k, v in results.items() if k != "random" and v["recall@3"] < rand + 0.25]
save_json({"model_accuracy": float(acc), "results": results, "failed": failed},
          res_dir / "sanity_check.json")
if failed:
    print(f"\nSANITY CHECK FAILED for {failed}: they do not beat random by 0.25.")
    raise SystemExit(1)
print("\nSANITY CHECK PASSED: every explainer clearly beats random on known-answer data.")
print("Next: python 04_train_models.py")
