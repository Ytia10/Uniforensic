"""N2 STEP 4 - Explain cases with the deployed federated model (N4).

For every N4 seed: Integrated Gradients explanations (predicted class) of
  - the agencies' own cases (N1 train split: what agencies summarise and share),
  - N1 test cases (never seen in training: the 'non-members'),
  - the attacker's pool (Step 3).
Also fixes the clipping bound C from the ATTACKER/PUBLIC pool, so the bound does
not depend on any agency's private data.
Output: data_n2/attr_s<seed>.npz
Run:  python n2_02_explanations.py      (about 3-5 minutes)
"""
import time

import numpy as np

from ufn1.common import banner
from ufn2.common import deployed_model, explain, load_config_n2, load_n1, load_n4, path_n2

cfg2 = load_config_n2()
cfg1, data, keys = load_n1()
cfg4, parts, mdir = load_n4()
sh = np.load(path_n2(cfg2, "data") / "shadow.npz")

for seed in cfg4["run_seeds"]:
    banner(f"Seed {seed}: explaining with fed_{cfg2['deployed']['method']}_a{cfg2['deployed']['alpha']}_s{seed}")
    t0 = time.time()
    model = deployed_model(cfg1, cfg2, cfg4, seed)
    out = {}
    for name, X in [("train", data["train"]["X"]), ("test", data["test"]["X"]), ("shadow", sh["X"])]:
        out[f"attr_{name}"], out[f"pred_{name}"] = explain(model, X, cfg1, cfg2, cfg1["seed"])
        print(f"  {name:6s}: {len(X)} cases   [{time.time() - t0:.0f}s]")
    norms = np.linalg.norm(np.abs(out["attr_shadow"]), axis=1)
    out["C"] = np.quantile(norms, cfg2["summary"]["clip_quantile"])
    print(f"  clipping bound C = {float(out['C']):.4f} (the {cfg2['summary']['clip_quantile']:.0%} "
          f"quantile of explanation size on the attacker's pool)")
    np.savez(path_n2(cfg2, "data") / f"attr_s{seed}.npz", **out)
print("\nDone. Next: python n2_03_release_attack.py")
