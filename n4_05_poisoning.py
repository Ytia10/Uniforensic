"""N4 STEP 7 - One malicious agency: what happens to accuracy AND explanations?

Attacks (by ONE of the 8 agencies; by default the one holding the most fraud cases):
  none          - honest run, to measure what each robust rule costs on its own
  label_flip    - an insider relabels its fraud cases as benign ("hide fraud")
  scaled_update - the agency sends the opposite of its real update, scaled x5
Aggregation rules compared (Flower's implementations):
  fedavg, fedmedian, fedtrimmedavg, krum
Output: models_n4/poison_*.pt, models_n4/registry_poisoning.json
Run:  python n4_05_poisoning.py      (about 3-6 minutes)
"""
import numpy as np

from ufn1.common import banner, derive_seed, load_json, save_json, sha256_file
from ufn1.schema import CLASSES
from ufn4.common import load_config_n4, load_n1, new_mlp, path_n4, save_model
from ufn4.fl import run_federated

cfg4 = load_config_n4()
pc = cfg4["poisoning"]
cfg1, data, keys = load_n1()
tr, va = data["train"], data["val"]
mdir = path_n4(cfg4, "models")
parts = load_json(path_n4(cfg4, "data") / "partitions.json")
registry = []
src, dst = CLASSES.index(pc["label_flip"]["from"]), CLASSES.index(pc["label_flip"]["to"])

banner(f"Poisoning: 1 of {cfg4['agencies']} agencies attacks (alpha {pc['alpha']})")
for seed in cfg4["run_seeds"]:
    agencies = parts[f"{pc['alpha']}|{seed}"]
    if pc["attacker"] == "most_fraud":
        attacker = int(np.argmax([(tr["y"][np.array(i)] == src).sum() for i in agencies]))
    else:
        attacker = int(pc["attacker"])
    share = len(agencies[attacker]) / len(tr["y"])
    print(f"  seed {seed}: attacker = agency {attacker} ({len(agencies[attacker])} cases, "
          f"{100 * share:.0f}% of all training cases)")
    for attack in pc["attacks"]:
        clients = []
        for a, i in enumerate(agencies):
            X, y = tr["X"][np.array(i)], tr["y"][np.array(i)].copy()
            if attack == "label_flip" and a == attacker:
                y[y == src] = dst
            clients.append((X, y))
        n_flip = int((tr["y"][np.array(agencies[attacker])] == src).sum()) if attack == "label_flip" else 0
        for rule in pc["aggregators"]:
            model, hist = run_federated(lambda: new_mlp(cfg1), clients, cfg1["mlp"], cfg4["federated"],
                                        derive_seed(seed, "poison", attack, rule), method="fedavg",
                                        rule=rule, attacker=attacker, attack=attack,
                                        attack_cfg=pc, pcfg=pc, val=(va["X"], va["y"]))
            f = mdir / f"poison_{attack}_{rule}_s{seed}.pt"
            save_model(model, f)
            registry.append({"file": f.name, "kind": "poison", "attack": attack, "rule": rule,
                             "alpha": pc["alpha"], "seed": seed, "attacker": attacker,
                             "attacker_share": share, "flipped_cases": n_flip,
                             "sha256": sha256_file(f)})
            print(f"  seed {seed}  {attack:13s} {rule:13s} final val macro-F1 {hist[-1]:.3f}"
                  + (f"   ({n_flip} fraud cases relabelled)" if n_flip and rule == pc["aggregators"][0] else ""))
save_json(registry, mdir / "registry_poisoning.json")
print(f"\nSaved {len(registry)} models. Next: python n4_06_score_models.py")
