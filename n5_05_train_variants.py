"""STEP 7 - Train one triage model per variant, foreign-residue prior and seed.

Same MLP, same training code and same hyper-parameters as N1 (ufn1.models.train_mlp,
config.yaml 'mlp'); only the INPUT differs between variants. Early stopping uses
val_fit (half of N1's validation cases); val_cal is kept for conformal calibration.

  rho = 0.0  : benign training cases are all clean (the analyst never saw residue)
  rho = 0.25 : a quarter of benign training cases carry foreign residue - a model
               that has learned that recovered-only evidence is sometimes not the
               suspect's. Only a provenance-aware input (C) can use that lesson.

Output: models_n5/{V}_rho{rho}_s{seed}.pt, results_n5/training.csv
Run:  python n5_05_train_variants.py      (about 5-15 min)
"""
import time
from itertools import product
from multiprocessing import Pool

import pandas as pd
import torch

from ufn1.common import banner, load_config
from ufn1.models import mlp_predict_proba, train_mlp
from ufn1.schema import CLASSES
from ufn5.common import load_config_n5, load_split, path_n5
from ufn5.sets import load, select, val_split_cases
from sklearn.metrics import f1_score

cfg1, cfg5 = load_config(), load_config_n5()
ddir, mdir, rdir = path_n5(cfg5, "data"), path_n5(cfg5, "models"), path_n5(cfg5, "results")
SEED = cfg5["seed"]
TR, VA = load(ddir, "train"), load(ddir, "val")
FIT_CASES, _ = val_split_cases(load_split("val")[0])


def work(job):
    v, rho, seed = job
    torch.set_num_threads(1)
    t0 = time.time()
    itr = select(TR, rho, SEED)
    iva = select(VA, rho, SEED, cases=FIT_CASES)
    Xtr, ytr = TR[f"X_{v}"][itr], TR["y"][itr]
    Xva, yva = VA[f"X_{v}"][iva], VA["y"][iva]
    model = train_mlp(Xtr, ytr, Xva, yva, cfg1["mlp"], len(CLASSES), seed, verbose=False)
    torch.save(model.state_dict(), mdir / f"{v}_rho{rho}_s{seed}.pt")
    f1 = f1_score(yva, mlp_predict_proba(model, Xva).argmax(1), average="macro")
    return {"variant": v, "rho": rho, "seed": seed, "n_inputs": Xtr.shape[1], "n_train": len(itr),
            "n_val_fit": len(iva), "val_fit_macro_f1": float(f1), "seconds": time.time() - t0}


if __name__ == "__main__":
    banner("Train triage models: variants x foreign prior x seeds")
    jobs = list(product(cfg5["variants"], cfg5["foreign_rate_train"], cfg5["train_seeds"]))
    rows = []
    with Pool(cfg5["workers"]) as pool:
        for r in pool.imap(work, jobs):
            rows.append(r)
            print(f"  {r['variant']:3s} rho={r['rho']:<4} seed={r['seed']}  inputs={r['n_inputs']}  "
                  f"val_fit macro-F1={r['val_fit_macro_f1']:.3f}  ({r['seconds']:.0f}s)")
    pd.DataFrame(rows).to_csv(rdir / "training.csv", index=False)
    print("\nDone. Next: python n5_06_evaluate.py")
