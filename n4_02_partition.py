"""N4 STEP 4 - Split N1's 2,400 training cases across 8 agencies (non-IID).

For every alpha in config_n4.yaml and every run seed, draw a Dirichlet split.
Test (600) and validation (600) cases stay exactly as in N1.
Output: data_n4/partitions.json, results_n4/partition_counts.csv
Run:  python n4_02_partition.py
"""
import pandas as pd

from ufn1.common import banner, derive_seed, save_json
from ufn1.schema import CLASSES
from ufn4.common import load_config_n4, load_n1, path_n4
from ufn4.partition import class_counts, dirichlet_partition

cfg4 = load_config_n4()
cfg1, data, keys = load_n1()
y = data["train"]["y"]
parts_all, rows = {}, []
for alpha in cfg4["alphas"]:
    for seed in cfg4["run_seeds"]:
        parts = dirichlet_partition(y, cfg4["agencies"], alpha, derive_seed(seed, "partition", alpha),
                                    cfg4["min_cases_per_agency"])
        parts_all[f"{alpha}|{seed}"] = [p.tolist() for p in parts]
        for a, row in enumerate(class_counts(parts, y, len(CLASSES))):
            rows.append({"alpha": alpha, "seed": seed, "agency": a, "total": int(row.sum()),
                         **{c: int(n) for c, n in zip(CLASSES, row)}})
save_json(parts_all, path_n4(cfg4, "data") / "partitions.json")
df = pd.DataFrame(rows)
df.to_csv(path_n4(cfg4, "results") / "partition_counts.csv", index=False)

s0 = cfg4["run_seeds"][0]
for alpha in cfg4["alphas"]:
    banner(f"alpha = {alpha}  (seed {s0}): cases per agency and class")
    print(df[(df.alpha == alpha) & (df.seed == s0)][["agency", "total"] + CLASSES].to_string(index=False))
print("\nDone. Next: python n4_03_train_baselines.py")
