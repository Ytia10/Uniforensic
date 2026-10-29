"""N4 STEP 5 - Train the two baselines every federated result is compared with.

  centralised - all 2,400 cases pooled in one place (the upper bound; not allowed
                in reality because agencies cannot share raw evidence).
                Same recipe as N1, so seed 2026 reproduces N1's model exactly.
  local-only  - every agency trains alone on its own cases (the lower bound;
                what agencies do today). 20% of its cases are kept for early stopping.
Output: models_n4/central_*.pt, models_n4/local_*.pt, models_n4/registry_baselines.json
Run:  python n4_03_train_baselines.py      (about 2-4 minutes)
"""
import numpy as np
import torch

from ufn1.common import ROOT, banner, derive_seed, load_json, save_json, sha256_file
from ufn1.models import mlp_predict_proba, train_mlp
from ufn1.schema import CLASSES
from ufn4.common import load_config_n4, load_n1, path_n4, save_model

cfg4 = load_config_n4()
cfg1, data, keys = load_n1()
tr, va = data["train"], data["val"]
mdir = path_n4(cfg4, "models")
parts = load_json(path_n4(cfg4, "data") / "partitions.json")
registry = []

banner("Centralised models (one per seed)")
for seed in cfg4["run_seeds"]:
    m = train_mlp(tr["X"], tr["y"], va["X"], va["y"], cfg1["mlp"], len(CLASSES), seed, verbose=False)
    f = mdir / f"central_s{seed}.pt"
    save_model(m, f)
    registry.append({"file": f.name, "kind": "central", "seed": seed})
    acc = (mlp_predict_proba(m, data["test"]["X"]).argmax(1) == data["test"]["y"]).mean()
    print(f"  seed {seed}: test accuracy {acc:.3f}")
n1_model = torch.load(ROOT / cfg1["paths"]["models"] / "mlp.pt", weights_only=True)
same = all(torch.equal(n1_model[k], torch.load(mdir / f"central_s{cfg1['seed']}.pt", weights_only=True)[k])
           for k in n1_model)
print(f"  centralised seed {cfg1['seed']} identical to N1's model: {same}")

banner("Local-only models (one per agency, alpha and seed)")
for key, agencies in parts.items():
    alpha, seed = float(key.split("|")[0]), int(key.split("|")[1])
    for a, idx in enumerate(agencies):
        idx = np.array(idx)
        rng = np.random.default_rng(derive_seed(seed, "holdout", alpha, a))
        rng.shuffle(idx)
        n_ho = max(1, int(round(cfg4["local_only"]["holdout"] * len(idx))))
        ho, own = idx[:n_ho], idx[n_ho:]
        m = train_mlp(tr["X"][own], tr["y"][own], tr["X"][ho], tr["y"][ho], cfg1["mlp"], len(CLASSES),
                      derive_seed(seed, "local", alpha, a), verbose=False)
        f = mdir / f"local_a{alpha}_s{seed}_ag{a}.pt"
        save_model(m, f)
        counts = {c: int((tr["y"][own] == i).sum()) for i, c in enumerate(CLASSES)}
        registry.append({"file": f.name, "kind": "local", "alpha": alpha, "seed": seed, "agency": a,
                         "train_counts": counts})
    print(f"  alpha {alpha:<5} seed {seed}: {len(agencies)} local models")
for r in registry:
    r["sha256"] = sha256_file(mdir / r["file"])
save_json(registry, mdir / "registry_baselines.json")
print(f"\nSaved {len(registry)} models. Next: python n4_04_train_federated.py")
