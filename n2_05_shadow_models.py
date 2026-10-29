"""N2 STEP 7 - Train the attacker's shadow models and the positive-control model.

Per-case attacks (Step 8) need to know what an explanation looks like for a case a
model did NOT train on. The attacker learns this from SHADOW models it trains
itself on its own pool (Step 3), with a recipe like the target's:
  standard - 2,400 cases, N1 recipe (for the centralised and federated targets)
  small    - 300 cases, same recipe (for the agencies' local-only models)
  overfit  - 200 cases, no dropout, no weight decay, 300 epochs (for the control)
Positive control: a model trained with the 'overfit' recipe on 200 of N1's cases.
It memorises its training cases, so an attack that is implemented correctly MUST
find leakage there. If it does not, the attack code is wrong.
Output: models_n2/*.pt, models_n2/registry.json
Run:  python n2_05_shadow_models.py      (about 1-2 minutes)
"""
import copy

import numpy as np

from ufn1.common import banner, derive_seed, save_json, sha256_file
from ufn1.models import TriageMLP, mlp_predict_proba, train_mlp
from ufn1.schema import CLASSES, FEATURES
from ufn2.common import load_config_n2, load_n1, load_n4, path_n2
from ufn4.common import save_model
from ufn4.fl import local_train

cfg2 = load_config_n2()
mc = cfg2["case_mia"]
cfg1, data, keys = load_n1()
cfg4, parts, mdir = load_n4()
seed = cfg4["run_seeds"][cfg2["analysis_seed_index"]]
sh = np.load(path_n2(cfg2, "data") / "shadow.npz")
out_dir = path_n2(cfg2, "models")
oc = mc["overfit_control"]
cfg_over = copy.deepcopy(cfg1["mlp"])
cfg_over["weight_decay"] = 0.0
NF, NC = len(FEATURES), len(CLASSES)


def train_recipe(recipe: str, X, y, s: int):
    """Train one model with the named recipe; returns (model, indices actually trained on)."""
    rng = np.random.default_rng(s)
    if recipe == "overfit":
        m = TriageMLP(NF, NC, cfg1["mlp"]["hidden"], 0.0)
        local_train(m, X, y, oc["epochs"], cfg_over, s)
        return m, np.arange(len(y))
    hold = 0.1 if recipe == "standard" else cfg4["local_only"]["holdout"]
    order = rng.permutation(len(y))
    n_ho = max(1, int(round(hold * len(y))))
    ho, own = order[:n_ho], order[n_ho:]
    m = train_mlp(X[own], y[own], X[ho], y[ho], cfg1["mlp"], NC, s, verbose=False)
    return m, own


registry = []
sizes = {"standard": len(data["train"]["y"]), "small": 300, "overfit": oc["cases"]}
for recipe, n in sizes.items():
    banner(f"Shadow recipe '{recipe}': {mc['shadows_per_recipe']} models on {n} attacker cases each")
    for k in range(mc["shadows_per_recipe"]):
        s = derive_seed(seed, "shadow", recipe, k)
        pick = np.random.default_rng(s).choice(len(sh["y"]), size=n, replace=False)
        model, own = train_recipe(recipe, sh["X"][pick], sh["y"][pick], s)
        f = out_dir / f"shadow_{recipe}_{k}.pt"
        save_model(model, f)
        registry.append({"file": f.name, "kind": "shadow", "recipe": recipe, "k": k,
                         "members": pick[own].tolist(), "sha256": sha256_file(f)})
    print(f"  {mc['shadows_per_recipe']} models trained")

banner(f"Positive control: overfit model on {oc['cases']} of N1's training cases")
s = derive_seed(seed, "overfit_control")
pick = np.random.default_rng(s).choice(len(data["train"]["y"]), size=oc["cases"], replace=False)
model, _ = train_recipe("overfit", data["train"]["X"][pick], data["train"]["y"][pick], s)
f = out_dir / "overfit_control.pt"
save_model(model, f)
acc_tr = (mlp_predict_proba(model, data["train"]["X"][pick]).argmax(1) == data["train"]["y"][pick]).mean()
acc_te = (mlp_predict_proba(model, data["test"]["X"]).argmax(1) == data["test"]["y"]).mean()
print(f"  accuracy on its own training cases {acc_tr:.3f} vs unseen test cases {acc_te:.3f}"
      f"  (a large gap = memorisation, as intended)")
registry.append({"file": f.name, "kind": "overfit_control", "members": pick.tolist(), "sha256": sha256_file(f)})
save_json(registry, out_dir / "registry.json")
print("\nDone. Next: python n2_06_case_mia.py")
