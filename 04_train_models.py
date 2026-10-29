"""STEP 6 - Train the triage models.

  MLP     - the model we explain with KernelSHAP / IG / GradientSHAP / LIME / Ablation
  XGBoost - a second model whose EXACT TreeSHAP gives an independent reference

Output: models/mlp.pt, models/xgb.json, results/model_performance.json
Run:  python 04_train_models.py
"""
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

from ufn1.common import (banner, get_path, load_config, load_json, save_json,
                         set_all_seeds, sha256_file)
from ufn1.models import mlp_predict_proba, train_mlp, train_xgb
from ufn1.schema import CLASSES

cfg = load_config()
seed = cfg["seed"]
set_all_seeds(seed)
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))
tr, va, te = (np.load(data_dir / f"{s}.npz") for s in ("train", "val", "test"))
tiers = np.array([c["tier"] for c in load_json(data_dir / "test_answer_key.json")])

banner("Training MLP")
mlp = train_mlp(tr["X"], tr["y"], va["X"], va["y"], cfg["mlp"], len(CLASSES), seed)
torch.save(mlp.state_dict(), model_dir / "mlp.pt")

banner("Training XGBoost")
xgb = train_xgb(tr["X"], tr["y"], cfg["xgboost"], seed)
xgb.save_model(model_dir / "xgb.json")

perf = {}
for name, proba in [("mlp", mlp_predict_proba(mlp, te["X"])), ("xgb", xgb.predict_proba(te["X"]))]:
    pred = proba.argmax(1)
    perf[name] = {
        "macro_f1": float(f1_score(te["y"], pred, average="macro")),
        "per_class_f1": dict(zip(CLASSES, f1_score(te["y"], pred, average=None).round(4).tolist())),
        "accuracy_by_tier": {t: float((pred[tiers == t] == te["y"][tiers == t]).mean())
                             for t in cfg["benchmark"]["tiers"]},
    }
    banner(f"{name.upper()} on the TEST split")
    print(classification_report(te["y"], pred, target_names=CLASSES, digits=3))
    print("  accuracy by tier:", {k: round(v, 3) for k, v in perf[name]["accuracy_by_tier"].items()})

perf["model_hashes"] = {"mlp.pt": sha256_file(model_dir / "mlp.pt"),
                        "xgb.json": sha256_file(model_dir / "xgb.json")}
save_json(perf, res_dir / "model_performance.json")
print("\nDone. Next: python 05_oracle_check.py")
