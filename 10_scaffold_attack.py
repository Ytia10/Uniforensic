"""STEP 12 (OPTIONAL, answers RQ4) - Can the acceptance test catch a model that
HIDES its real reasoning from the explainers?

Slack et al. (AIES 2020, 'Fooling LIME and SHAP') showed that perturbation-based
explainers query the model on unnatural (perturbed) inputs. A 'scaffolded' model
detects those inputs and answers them with an innocent model, while real cases
get the real (biased) model. Explanations then blame an innocent feature.

Our scaffold, adapted to tabular forensic features:
  real model      = the trained MLP (uses the planted evidence)
  innocuous model = depends ONLY on web_social_share (unrelated to the offence)
  OOD detector    = gradient-boosted classifier: real training rows vs perturbed rows
                    (features replaced by background values or by 0, like
                     KernelSHAP / LIME / ablation do)
Measured: does recall@|G| collapse for each explainer, and does the case-level
agreement check (Level 2, criterion b) flag the fooled explanations?
F-Fidelity is not applied here because the scaffold cannot be fine-tuned.

Output: results/scaffold_attack.json, results/scaffold_attack_per_case.csv
Run:  python 10_scaffold_attack.py        (about 5-10 minutes; uses 50 cases per tier)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

from ufn1.common import banner, derive_seed, get_path, load_config, load_json, save_json, set_all_seeds
from ufn1.explainers import run_explainer
from ufn1.metrics import feature_agreement, recall_at_k, topk_idx
from ufn1.models import TriageMLP, mlp_predict_proba
from ufn1.schema import CLASSES, FEATURES

cfg = load_config()
seed = cfg["seed"]
set_all_seeds(seed)
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))
tr, te = np.load(data_dir / "train.npz"), np.load(data_dir / "test.npz")
keys = load_json(data_dir / "test_answer_key.json")
A = np.load(res_dir / "attributions_mlp.npz")
oracle = {o["case_index"]: o["passes"] for o in load_json(res_dir / "oracle_cases.json") if o["model"] == "mlp"}

mlp = TriageMLP(len(FEATURES), len(CLASSES), cfg["mlp"]["hidden"], cfg["mlp"]["dropout"])
mlp.load_state_dict(torch.load(model_dir / "mlp.pt", weights_only=True))
mlp.eval()
INNOCENT = FEATURES.index("web_social_share")
rng = np.random.default_rng(derive_seed(seed, "scaffold"))

# ----------------------------------------------------- 1. the OOD detector
banner("1. Train the out-of-distribution detector")
Xr = tr["X"]
n, d = Xr.shape
pert = Xr.copy()
for i in range(n):
    m = rng.random(d) < rng.uniform(0.1, 0.9)                    # which features get replaced
    pert[i, m] = 0.0 if rng.random() < 0.5 else Xr[rng.integers(n), m]
ood = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
ood.fit(np.vstack([Xr, pert]), np.r_[np.ones(n), np.zeros(n)])
print(f"  detector says 'real' for {ood.predict(te['X']).mean():.3f} of real TEST cases")


# --------------------------------------------------------- 2. the scaffold
class Innocuous(nn.Module):
    """Class scores depend only on web_social_share."""
    def forward(self, x):
        v = x[:, INNOCENT:INNOCENT + 1]
        return torch.cat([-2 * v, 2 * v, 3 * v, -3 * v], dim=1)


class Scaffold(nn.Module):
    def __init__(self, real, fake, detector):
        super().__init__()
        self.real, self.fake, self.detector = real, fake, detector

    def forward(self, x):
        is_real = self.detector.predict(x.detach().numpy())     # not differentiable: a switch
        s = torch.tensor(is_real, dtype=x.dtype).unsqueeze(1)
        return s * self.real(x) + (1 - s) * self.fake(x)


scaffold = Scaffold(mlp, Innocuous(), ood).eval()
same = (mlp_predict_proba(scaffold, te["X"]).argmax(1) == mlp_predict_proba(mlp, te["X"]).argmax(1)).mean()
print(f"  scaffold gives the SAME prediction as the real model on {same:.3f} of test cases")

# ------------------------------------------------------ 3. explain the scaffold
banner("2. Explain the scaffold on oracle-passed offence cases")
# 50 oracle-passed cases per tier (KernelSHAP on a scaffold is slow: every
# perturbed row also goes through the detector)
n_per_tier = cfg.get("scaffold", {}).get("cases_per_tier", 50)
sel = []
for t in cfg["benchmark"]["tiers"]:
    cand = [j for j, i in enumerate(A["case_index"]) if oracle[int(i)] and keys[int(i)]["tier"] == t]
    sel += list(rng.choice(cand, size=min(n_per_tier, len(cand)), replace=False))
sel = np.array(sorted(sel))
idx = A["case_index"][sel]
X, y = te["X"][idx], te["y"][idx]
background = tr["X"][A["background_rows"]]
names = ["kernelshap", "lime", "integrated_gradients", "feature_ablation"]
att = {e: run_explainer(e, scaffold, X, y, background, cfg["explainers"][e], seed) for e in names}

# ------------------------------------------------------------- 4. measure
banner("3. Did the attack work, and did the acceptance check notice?")
G = [set(keys[i]["primary_idx"]) for i in idx]
summary = {"ood_real_rate_test": float(ood.predict(te["X"]).mean()), "prediction_match": float(same),
           "n_cases": int(len(idx)), "explainers": {}}
for e in names:
    before = np.mean([recall_at_k(a, g, len(g)) for a, g in zip(A[e][sel], G)])
    after = np.mean([recall_at_k(a, g, len(g)) for a, g in zip(att[e], G)])
    top1 = np.mean([topk_idx(a, 1)[0] == INNOCENT for a in att[e]])
    summary["explainers"][e] = {"recall_before": float(before), "recall_after": float(after),
                                "innocent_feature_top1": float(top1)}
    print(f"  {e:22s} recall@|G| {before:.3f} -> {after:.3f}   innocent feature ranked #1: {top1:.3f}")

P, S = cfg["acceptance"]["primary_explainer"], cfg["acceptance"]["secondary_explainer"]
k, thr = cfg["scoring"]["k_fixed"], cfg["acceptance"]["case_min_agreement"]
agree_att = np.array([feature_agreement(a, b, k) for a, b in zip(att[P], att[S])])
agree_clean = np.array([feature_agreement(a, b, k) for a, b in zip(A[P][sel], A[S][sel])])
fooled = np.array([recall_at_k(a, g, len(g)) < cfg["acceptance"]["good_explanation_recall"]
                   for a, g in zip(att[P], G)])
summary["agreement_check"] = {
    "flag_rate_clean_model": float((agree_clean < thr).mean()),
    "flag_rate_scaffold": float((agree_att < thr).mean()),
    "fooled_primary_explanations": int(fooled.sum()),
    "fooled_and_flagged": float((agree_att[fooled] < thr).mean()) if fooled.any() else None}
ac = summary["agreement_check"]
print(f"\n  {P} vs {S} agreement check (FA@{k} < {thr} => NOT VALIDATED):")
print(f"    flagged on the honest model : {ac['flag_rate_clean_model']:.3f}")
print(f"    flagged on the scaffold     : {ac['flag_rate_scaffold']:.3f}")
print(f"    fooled {P} explanations caught: {ac['fooled_and_flagged']}  (of {ac['fooled_primary_explanations']})")
save_json(summary, res_dir / "scaffold_attack.json")
pd.DataFrame({"case_index": idx, "agree_scaffold": agree_att, "agree_clean": agree_clean,
              "primary_fooled": fooled}).to_csv(res_dir / "scaffold_attack_per_case.csv", index=False)
print("\nDone. Next: python 11_make_report.py")
