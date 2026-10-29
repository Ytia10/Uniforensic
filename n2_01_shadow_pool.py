"""N2 STEP 3 - Give the attacker its OWN data (never N1's cases).

A realistic attacker knows how cases look in general, not the agencies' actual
cases. It runs the same public case generator as N1 with a different split name
('shadow'), so every case is a new random draw. Features use N1's scaler.
The attacker uses this pool to (a) estimate what an average explanation looks like,
(b) build practice aggregates for the distinguishability game, and (c) train
shadow models for the per-case attacks.
Output: data_n2/shadow.npz (X, y)
Run:  python n2_01_shadow_pool.py        (about 2-3 minutes)
"""
import time

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed, get_path, load_json
from ufn1.features import apply_scaler, extract_features
from ufn1.generator import generate_case
from ufn1.schema import ARTIFACT_TYPES, CLASSES
from ufn2.common import load_config_n2, load_n1, path_n2

cfg2 = load_config_n2()
cfg1, data, keys = load_n1()
bcfg = cfg1["benchmark"]
scaler = load_json(get_path(cfg1, "models") / "scaler.json")
n_per = cfg2["shadow_pool"]["n_per_class_tier"]

banner(f"Generating the attacker's pool: {n_per} cases per class x tier")
t0, X_parts, y = time.time(), [], []
idx = 0
for label in bcfg["classes"]:
    for tier in bcfg["tiers"]:
        rows = {t: [] for t in ARTIFACT_TYPES}
        ids = []
        for _ in range(n_per):
            cid = f"SH{idx:05d}"
            r, _key = generate_case(cid, label, tier, bcfg,
                                    np.random.default_rng(derive_seed(cfg1["seed"], "shadow", idx)))
            for t in ARTIFACT_TYPES:
                rows[t].extend(r[t])
            ids.append(cid)
            y.append(CLASSES.index(label))
            idx += 1
        tables = {t: pd.DataFrame(rows[t]) for t in ARTIFACT_TYPES}
        X_parts.append(apply_scaler(extract_features(tables, ids), scaler))
        print(f"  {label:10s} {tier:7s} done   [{time.time() - t0:.0f}s]")
X, y = np.vstack(X_parts), np.array(y)
np.savez(path_n2(cfg2, "data") / "shadow.npz", X=X, y=y)
print(f"\nSaved {len(y)} attacker cases. Next: python n2_02_explanations.py")
