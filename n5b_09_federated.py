"""N5b STEP 11 - N4 alignment: can agencies learn the provenance-aware model federatedly?

Every agency can compute the same 36 inputs of variant C from its own devices (the
27 N1 features + 9 recovered-channel features, all defined on the shared schema).
This step trains variants L, B and C with N4's FedAvg (ufn4.fl.run_federated, Flower's
aggregation code) on N4's own 8-agency non-IID partition (data_n4/partitions.json,
Dirichlet alpha = 0.5, seed 2026) and compares with the centralised models of step 7.

Output: models_n5b/fed_*.pt, results_n5b/federated.csv
Run:  python n5b_09_federated.py      (about 1-3 min)
"""
import numpy as np
import pandas as pd
import torch

from ufn1.common import ROOT, banner, derive_seed, load_config, load_json
from ufn1.models import TriageMLP, mlp_predict_proba
from ufn1.schema import CLASSES
from ufn4.common import load_config_n4
from ufn4.fl import run_federated
from ufn5.common import load_split, path_n5
from ufn5b.common import load_config_n5b
from ufn5.evaluate import load_mlp, triage_scores
from ufn5.sets import BENIGN, load, select, val_split_cases

cfg1, cfg4, cfg5 = load_config(), load_config_n4(), load_config_n5b()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
SEED, fcfg = cfg5["seed"], cfg5["federated"]
key = f"{fcfg['dirichlet_alpha']}|{fcfg['seed']}"
parts = load_json(ROOT / cfg4["paths"]["data"] / "partitions.json")[key]
TR, VA, TE = load(ddir, "train"), load(ddir, "val"), load(ddir, "test")
FIT_CASES, _ = val_split_cases(load_split("val")[0])
rho, seed = 0.25, fcfg["seed"]
rows = []

banner(f"FedAvg on N4's partition {key}: {len(parts)} agencies")
itr = select(TR, rho, SEED)
iva = select(VA, rho, SEED, cases=FIT_CASES)
for v in ("L", "B", "C"):
    X, y = TR[f"X_{v}"], TR["y"]
    clients = []
    for agency in parts:
        m = itr[np.isin(TR["case_idx"][itr], agency)]
        clients.append((X[m], y[m]))
    n_in = X.shape[1]
    model, hist = run_federated(
        lambda: TriageMLP(n_in, len(CLASSES), cfg1["mlp"]["hidden"], cfg1["mlp"]["dropout"]),
        clients, cfg1["mlp"], cfg4["federated"], derive_seed(seed, "fed_n5", v, rho),
        method="fedavg", val=(VA[f"X_{v}"][iva], VA["y"][iva]))
    torch.save(model.state_dict(), mdir / f"fed_{v}_rho{rho}_s{seed}.pt")
    central = load_mlp(cfg1, mdir / f"{v}_rho{rho}_s{seed}.pt", n_in)
    for name, mdl in (("federated", model), ("centralised", central)):
        pred = mlp_predict_proba(mdl, TE[f"X_{v}"]).argmax(1)
        for lv in cfg5["scenario"]["levels"]:
            idx = select(TE, 0.0, SEED, level=lv)
            rows.append({"variant": v, "training": name, "level": lv, **triage_scores(TE["y"][idx], pred[idx])})
        ben = np.where((TE["y"] == BENIGN) & TE["foreign"].astype(bool))[0]
        rows.append({"variant": v, "training": name, "level": "benign_residue",
                     "false_incrimination": float((pred[ben] != BENIGN).mean()), "n": len(ben)})
    d = pd.DataFrame(rows).query("variant == @v and level != 'benign_residue'")
    f = d.groupby("training").macro_f1.mean()
    print(f"  {v}: mean test macro-F1 over levels  federated {f['federated']:.3f}   "
          f"centralised {f['centralised']:.3f}   (clients: {[len(c[1]) for c in clients]})")
pd.DataFrame(rows).to_csv(rdir / "federated.csv", index=False)
print("\nDone. Next: python n5b_10_case_export.py")
