"""STEP 10 - Removal-based faithfulness done properly: F-Fidelity (ICLR 2025).

This is the ground-truth-FREE faithfulness check. It matters because on real
casework there is no planted answer key; step 11 uses it per case.

  1. fine-tune a copy of the MLP with random masking (beta = 0.1)  -> surrogate
  2. per case and explainer: FFid+ (remove top features, should HURT) and
     FFid- (remove bottom features, should NOT hurt)
  3. naive deletion on the ORIGINAL model as the baseline the paper improves on

Output: models/mlp_ffid_surrogate.pt, results/ffidelity_per_case.csv,
        results/table_ffidelity.csv
Run:  python 08_ffidelity.py
"""
import copy

import numpy as np
import pandas as pd
import torch

from ufn1.common import banner, get_path, load_config, load_json, set_all_seeds
from ufn1.explainers import EXPLAINERS
from ufn1.ffidelity import ffidelity_per_case, naive_deletion_per_case
from ufn1.metrics import bootstrap_ci
from ufn1.models import TriageMLP, mlp_predict_proba, train_mlp
from ufn1.schema import CLASSES, FEATURES

cfg = load_config()
seed, fcfg = cfg["seed"], cfg["ffidelity"]
set_all_seeds(seed)
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))
tr, va, te = (np.load(data_dir / f"{s}.npz") for s in ("train", "val", "test"))
keys = load_json(data_dir / "test_answer_key.json")
oracle = {o["case_index"]: o["passes"] for o in load_json(res_dir / "oracle_cases.json") if o["model"] == "mlp"}
A = np.load(res_dir / "attributions_mlp.npz")
idx = A["case_index"]

mlp = TriageMLP(len(FEATURES), len(CLASSES), cfg["mlp"]["hidden"], cfg["mlp"]["dropout"])
mlp.load_state_dict(torch.load(model_dir / "mlp.pt", weights_only=True))
mlp.eval()

banner(f"Fine-tuning the F-Fidelity surrogate (random masking, beta={fcfg['beta']})")
surrogate = train_mlp(tr["X"], tr["y"], va["X"], va["y"], cfg["mlp"], len(CLASSES), seed,
                      mask_beta=fcfg["beta"], init_model=copy.deepcopy(mlp),
                      max_epochs=fcfg["finetune_epochs"], verbose=False)
torch.save(surrogate.state_dict(), model_dir / "mlp_ffid_surrogate.pt")
acc_o = (mlp_predict_proba(mlp, te["X"]).argmax(1) == te["y"]).mean()
acc_s = (mlp_predict_proba(surrogate, te["X"]).argmax(1) == te["y"]).mean()
print(f"  test accuracy: original {acc_o:.3f}  surrogate {acc_s:.3f} (should be close)")

banner("Scoring every explainer")
X, y = te["X"][idx], te["y"][idx]
rows = []
for name in EXPLAINERS:
    ff, info = ffidelity_per_case(surrogate, X, y, A[name], fcfg["beta"], fcfg["alpha"],
                                  fcfg["n_samples"], fcfg["s_top"], seed)
    nd = naive_deletion_per_case(mlp, X, y, A[name], fcfg["s_top"])
    for j, i in enumerate(idx):
        rows.append({"explainer": name, "case_index": int(i), "case_id": keys[i]["case_id"],
                     "tier": keys[i]["tier"], "oracle_pass": oracle[int(i)],
                     **{m: v[j] for m, v in ff.items()}, "naive_deletion_drop": nd[j]})
    print(f"  {name:22s} removes {info['m_plus']} of top-{fcfg['s_top']} / {info['m_minus']} of the rest")
df = pd.DataFrame(rows)
df.to_csv(res_dir / "ffidelity_per_case.csv", index=False)

METRICS = ["ffid_plus", "ffid_minus", "ffid_plus_prob", "ffid_minus_prob", "naive_deletion_drop"]
ev = df[df.oracle_pass]
table = []
for (name, tier), g in list(ev.groupby(["explainer", "tier"])) + [((n, "ALL"), g) for n, g in ev.groupby("explainer")]:
    row = {"explainer": name, "tier": tier, "n_cases": len(g)}
    for m in METRICS:
        row[m], row[m + "_lo"], row[m + "_hi"] = bootstrap_ci(g[m].to_numpy(), cfg["scoring"]["bootstrap"], seed)
    table.append(row)
tab = pd.DataFrame(table)
tab.to_csv(res_dir / "table_ffidelity.csv", index=False)
print("\n  Pooled over tiers (FFid+ high = faithful, FFid- near 0 = faithful):")
for name, r in tab[tab.tier == "ALL"].set_index("explainer").reindex(EXPLAINERS).iterrows():
    print(f"    {name:22s} FFid+={r.ffid_plus:.3f} [{r.ffid_plus_lo:.3f},{r.ffid_plus_hi:.3f}]  "
          f"FFid-={r.ffid_minus:.3f} | prob: FFid+={r.ffid_plus_prob:.3f} FFid-={r.ffid_minus_prob:.3f}"
          f" | naive_drop={r.naive_deletion_drop:.3f}")
print("\nDone. Next: python 09_acceptance_test.py")
