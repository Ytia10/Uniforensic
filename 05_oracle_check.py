"""STEP 7 - Oracle check: does the model REALLY use the planted evidence?

A model can be right for the wrong reason (a shortcut). Scoring explanations
against planted evidence only makes sense if the model's decision actually
depends on that evidence. Two checks:

 7a Global  - retrain WITHOUT the 9 decisive features: macro-F1 must collapse.
              Retrain with ONLY them: macro-F1 should stay high.
 7b Per case - feed the case's counterfactual (same artifacts, planted evidence
              removed, features re-extracted - see step 4). The case PASSES if
              the model is correct AND P(true class) drops by >= oracle.min_prob_drop.
Only passing cases enter explanation scoring. Report the pass rate in the paper.

Output: results/oracle_global.json, results/oracle_cases.json
Run:  python 05_oracle_check.py
"""
import numpy as np
import torch
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from ufn1.common import banner, get_path, load_config, load_json, save_json, set_all_seeds
from ufn1.models import TriageMLP, mlp_predict_proba, train_mlp
from ufn1.schema import CLASSES, DECISIVE_FEATURES, FEATURES

cfg = load_config()
seed = cfg["seed"]
set_all_seeds(seed)
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))
tr, va, te = (np.load(data_dir / f"{s}.npz") for s in ("train", "val", "test"))
keys = load_json(data_dir / "test_answer_key.json")

mlp = TriageMLP(len(FEATURES), len(CLASSES), cfg["mlp"]["hidden"], cfg["mlp"]["dropout"])
mlp.load_state_dict(torch.load(model_dir / "mlp.pt", weights_only=True))
mlp.eval()
xgb = XGBClassifier()
xgb.load_model(model_dir / "xgb.json")

# ----------------------------------------------------------- 7a global ablation
banner("7a. Global ablation (retrain MLP on feature subsets)")
dec = [FEATURES.index(f) for f in DECISIVE_FEATURES]
rest = [i for i in range(len(FEATURES)) if i not in dec]
# "proxy" features: background features that planted artifacts ALSO change
# (e.g. scam SMS raise sms_out_total) in >= 20% of offence training cases
train_keys = [k for k in load_json(data_dir / "train_answer_key.json") if k["label"] != "benign"]
freq = {f: np.mean([f in k["affected_features"] for k in train_keys]) for f in FEATURES}
proxies = [f for f in FEATURES if f not in DECISIVE_FEATURES and freq[f] >= 0.2]
clean = [i for i in rest if FEATURES[i] not in proxies]
print(f"  proxy features (changed by planted evidence in >=20% of cases): {proxies}")
glob = {"all_features": float(f1_score(te["y"], mlp_predict_proba(mlp, te["X"]).argmax(1), average="macro")),
        "proxy_features": proxies}
for name, cols in [("without_decisive", rest), ("without_decisive_and_proxies", clean),
                   ("only_decisive", dec)]:
    m = train_mlp(tr["X"][:, cols], tr["y"], va["X"][:, cols], va["y"], cfg["mlp"],
                  len(CLASSES), seed, verbose=False)
    glob[name] = float(f1_score(te["y"], mlp_predict_proba(m, te["X"][:, cols]).argmax(1), average="macro"))
for k, v in glob.items():
    if k != "proxy_features":
        print(f"  MLP macro-F1 {k:30s} = {v:.3f}")
print("  (chance level for 4 balanced classes is about 0.25)")
save_json(glob, res_dir / "oracle_global.json")

# --------------------------------------------------------- 7b per-case oracle
banner("7b. Per-case counterfactual oracle")
thr = cfg["oracle"]["min_prob_drop"]
y = te["y"]
rows = []
for model_name, proba_fn in [("mlp", lambda X: mlp_predict_proba(mlp, X)),
                             ("xgb", lambda X: xgb.predict_proba(X))]:
    p_full, p_cf = proba_fn(te["X"]), proba_fn(te["X_cf"])
    for i, k in enumerate(keys):
        if k["label"] == "benign":
            continue
        correct = bool(p_full[i].argmax() == y[i])
        drop = float(p_full[i, y[i]] - p_cf[i, y[i]])
        rows.append({"case_index": i, "case_id": k["case_id"], "label": k["label"], "tier": k["tier"],
                     "model": model_name, "correct": correct, "prob_drop": drop,
                     "passes": bool(correct and drop >= thr)})

save_json(rows, res_dir / "oracle_cases.json")
print(f"  pass rule: correct AND P(true) drops by >= {thr} without the planted evidence\n")
print(f"  {'model':5s} {'tier':7s} {'cases':>6s} {'correct':>8s} {'pass':>6s} {'pass%':>6s}")
for model_name in ["mlp", "xgb"]:
    for tier in cfg["benchmark"]["tiers"] + ["ALL"]:
        r = [x for x in rows if x["model"] == model_name and (tier == "ALL" or x["tier"] == tier)]
        nc, npass = sum(x["correct"] for x in r), sum(x["passes"] for x in r)
        print(f"  {model_name:5s} {tier:7s} {len(r):6d} {nc:8d} {npass:6d} {100 * npass / len(r):5.1f}%")
print("\nDone. Next: python 06_run_explainers.py")
