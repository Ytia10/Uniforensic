"""N4 STEP 6 - Federated training: FedAvg and FedProx for every alpha and seed.

The 8 agencies never share cases, only model parameters (ufn4/fl.py).
The validation macro-F1 after every round is recorded for the convergence plot.
Output: models_n4/fed_*.pt, models_n4/registry_federated.json, results_n4/convergence.json
Run:  python n4_04_train_federated.py      (about 2-5 minutes)
"""
import numpy as np

from ufn1.common import banner, derive_seed, load_json, save_json, sha256_file
from ufn4.common import load_config_n4, load_n1, new_mlp, path_n4, save_model
from ufn4.fl import run_federated

cfg4 = load_config_n4()
cfg1, data, keys = load_n1()
tr, va = data["train"], data["val"]
mdir = path_n4(cfg4, "models")
parts = load_json(path_n4(cfg4, "data") / "partitions.json")
registry, conv = [], {}

banner(f"Federated training: {cfg4['agencies']} agencies, {cfg4['federated']['rounds']} rounds")
for key, agencies in parts.items():
    alpha, seed = float(key.split("|")[0]), int(key.split("|")[1])
    clients = [(tr["X"][np.array(i)], tr["y"][np.array(i)]) for i in agencies]
    for method in cfg4["federated"]["methods"]:
        model, hist = run_federated(lambda: new_mlp(cfg1), clients, cfg1["mlp"], cfg4["federated"],
                                    derive_seed(seed, "fed", method, alpha), method=method,
                                    val=(va["X"], va["y"]))
        f = mdir / f"fed_{method}_a{alpha}_s{seed}.pt"
        save_model(model, f)
        registry.append({"file": f.name, "kind": "federated", "method": method, "alpha": alpha,
                         "seed": seed, "sha256": sha256_file(f)})
        conv[f"{method}|{alpha}|{seed}"] = hist
        r90 = next((i + 1 for i, v in enumerate(hist) if v >= 0.9 * max(hist)), None)
        print(f"  {method:8s} alpha {alpha:<5} seed {seed}: final val macro-F1 {hist[-1]:.3f}"
              f"  (90% of best reached at round {r90})")
save_json(registry, mdir / "registry_federated.json")
save_json(conv, path_n4(cfg4, "results") / "convergence.json")
print(f"\nSaved {len(registry)} models. Next: python n4_05_poisoning.py")
