"""STEP 8 - Explain every offence case of the TEST split with every explainer.

We explain P(PREDICTED class) - what a real report would show, since the true
label is unknown on real casework - for all 450 offence test cases (benign cases
have no planted evidence). Step 9 scores only oracle-passed cases (there the
prediction is correct); all are kept so step 11 can judge the acceptance test.

Output: results/attributions_mlp.npz  (one array per explainer, shape n x 27)
        results/attributions_val.npz  (validation split, for calibration in step 11)
        results/attributions_xgb.npz  (exact TreeSHAP of the XGBoost reference)
Run:  python 06_run_explainers.py        (about 3-10 minutes on a laptop CPU)
"""
import numpy as np
import torch
from xgboost import XGBClassifier

from ufn1.common import banner, derive_seed, get_path, load_config, load_json, set_all_seeds
from ufn1.explainers import EXPLAINERS, run_explainer
from ufn1.models import TriageMLP, mlp_predict_proba, xgb_treeshap
from ufn1.schema import CLASSES, FEATURES

cfg = load_config()
seed = cfg["seed"]
set_all_seeds(seed)
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))
tr, te = np.load(data_dir / "train.npz"), np.load(data_dir / "test.npz")
keys = load_json(data_dir / "test_answer_key.json")

mlp = TriageMLP(len(FEATURES), len(CLASSES), cfg["mlp"]["hidden"], cfg["mlp"]["dropout"])
mlp.load_state_dict(torch.load(model_dir / "mlp.pt", weights_only=True))
mlp.eval()

idx = np.array([i for i, k in enumerate(keys) if k["label"] != "benign"])
X = te["X"][idx]
y = mlp_predict_proba(mlp, X).argmax(1)            # explain the PREDICTED class
bg_rows = np.random.default_rng(derive_seed(seed, "background")).choice(
    len(tr["X"]), cfg["explainers"]["kernelshap"]["background_size"], replace=False)
background = tr["X"][bg_rows]
print(f"Explaining {len(idx)} offence cases, {len(FEATURES)} features each")

banner("MLP explainers")
out = {"case_index": idx, "background_rows": bg_rows, "target": y}
for name in EXPLAINERS:
    out[name] = run_explainer(name, mlp, X, y, background, cfg["explainers"][name], seed)
np.savez(res_dir / "attributions_mlp.npz", **out)

banner("Validation split: primary + secondary explainer only (to calibrate step 11)")
va = np.load(data_dir / "val.npz")
vkeys = load_json(data_dir / "val_answer_key.json")
vidx = np.array([i for i, k in enumerate(vkeys) if k["label"] != "benign"])
vy = mlp_predict_proba(mlp, va["X"][vidx]).argmax(1)
vout = {"case_index": vidx, "target": vy}
for name in [cfg["acceptance"]["primary_explainer"], cfg["acceptance"]["secondary_explainer"]]:
    vout[name] = run_explainer(name, mlp, va["X"][vidx], vy, background,
                               cfg["explainers"][name], seed)
np.savez(res_dir / "attributions_val.npz", **vout)

banner("XGBoost exact TreeSHAP (reference model)")
xgb = XGBClassifier()
xgb.load_model(model_dir / "xgb.json")
ts = xgb_treeshap(xgb, X)
y_xgb = xgb.predict_proba(X).argmax(1)              # XGBoost's own prediction
np.savez(res_dir / "attributions_xgb.npz", case_index=idx, target=y_xgb,
         treeshap=ts[np.arange(len(y_xgb)), y_xgb, :])
print("  treeshap done")
print("\nDone. Next: python 07_score_explanations.py")
