"""STEP 3 - Generate the planted-evidence benchmark.

Output (folder data/):
  {split}_{sms|call|web|login|txn}.csv.gz   all artifacts in the unified schema
  {split}_answer_key.json                    planted evidence per case
Also writes results/frozen_thresholds.json (hash of the acceptance thresholds,
recorded BEFORE any model or explanation exists).
Run:  python 01_generate_benchmark.py
"""
import time

import numpy as np
import pandas as pd

from ufn1.common import (banner, derive_seed, get_path, load_config, save_json,
                         sha256_of, set_all_seeds)
from ufn1.generator import generate_case
from ufn1.schema import ARTIFACT_TYPES

cfg = load_config()
set_all_seeds(cfg["seed"])
bcfg = cfg["benchmark"]
data_dir, res_dir = get_path(cfg, "data"), get_path(cfg, "results")

# --- 1. freeze the acceptance thresholds before anything else exists -------
frozen = {"acceptance": cfg["acceptance"], "sha256": sha256_of(cfg["acceptance"]),
          "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
save_json(frozen, res_dir / "frozen_thresholds.json")
print(f"Acceptance thresholds frozen, sha256 = {frozen['sha256'][:16]}...")

# --- 2. generate every split ------------------------------------------------
for split, n_per in bcfg["n_per_class_tier"].items():
    banner(f"Generating split '{split}'  ({n_per} cases per class x tier)")
    t0 = time.time()
    rows = {t: [] for t in ARTIFACT_TYPES}
    keys = []
    idx = 0
    for label in bcfg["classes"]:
        for tier in bcfg["tiers"]:
            for _ in range(n_per):
                case_id = f"{split[:2].upper()}{idx:05d}"
                rng = np.random.default_rng(derive_seed(cfg["seed"], split, idx))
                r, key = generate_case(case_id, label, tier, bcfg, rng)
                for t in ARTIFACT_TYPES:
                    rows[t].extend(r[t])
                keys.append(key)
                idx += 1
    total = 0
    for t in ARTIFACT_TYPES:
        df = pd.DataFrame(rows[t])
        df.to_csv(data_dir / f"{split}_{t}.csv.gz", index=False, compression="gzip")
        total += len(df)
        n_ev = int((df.role == "evidence").sum())
        print(f"  {t:6s}: {len(df):8d} artifacts  ({n_ev} planted evidence)")
    save_json(keys, data_dir / f"{split}_answer_key.json")
    print(f"  {len(keys)} cases, {total} artifacts in {time.time() - t0:.1f}s")

print("\nDone. Next: python 02_build_features.py")
