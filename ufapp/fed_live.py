"""Live federated training (N4) with round-by-round progress.

The round loop below is ufn4.fl.run_federated() unrolled so the page can show each
round; it calls the same N4 primitives (get_params, set_params, local_train, combine
= Flower's aggregation code) in the same order with the same seeds. The run is then
compared with the model N4 stored for the same setting: identical parameters prove
the live demo IS the N4 experiment, not a re-implementation of it.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from ufn1.common import ROOT, derive_seed, load_json
from ufn1.models import mlp_predict_proba
from ufn1.schema import CLASSES
from ufn4.common import load_config_n4, load_model, load_n1, new_mlp
from ufn4.evaluate import score_model
from ufn4.fl import combine, get_params, local_train, set_params


def run(job, alpha: float, seed: int, method: str = "fedavg", attack: str = "none",
        rule: str = "fedavg", score: bool = True) -> dict:
    cfg4 = load_config_n4()
    cfg1, data, keys = load_n1()
    tr, va, te = data["train"], data["val"], data["test"]
    pc, fcfg = cfg4["poisoning"], cfg4["federated"]
    parts = load_json(ROOT / cfg4["paths"]["data"] / "partitions.json")
    key = f"{alpha}|{seed}"
    if key not in parts:
        raise ValueError(f"N4 has no partition for alpha={alpha}, seed={seed}")
    agencies = [np.array(i) for i in parts[key]]
    poisoned = attack != "none"
    if poisoned and (method != "fedavg" or float(alpha) != float(pc["alpha"])):
        raise ValueError(f"N4's poisoning runs use FedAvg updates at alpha {pc['alpha']}")
    if not poisoned and rule != method:
        raise ValueError("an honest run aggregates with its own method's rule")

    job.step("Split the training cases over the agencies (N4's stored Dirichlet partition)")
    src, dst = CLASSES.index(pc["label_flip"]["from"]), CLASSES.index(pc["label_flip"]["to"])
    attacker = None
    if poisoned:
        attacker = (int(np.argmax([(tr["y"][i] == src).sum() for i in agencies]))
                    if pc["attacker"] == "most_fraud" else int(pc["attacker"]))
    clients = []
    for a, i in enumerate(agencies):
        X, y = tr["X"][i], tr["y"][i].copy()
        if attack == "label_flip" and a == attacker:
            y[y == src] = dst
        clients.append((X, y))
    counts = [[int((c[1] == k).sum()) for k in range(len(CLASSES))] for c in clients]
    job.done_step(f"{len(agencies)} agencies, {sum(len(c[1]) for c in clients)} cases"
                  + (f"; agency {attacker} is malicious ({attack})" if poisoned else ""))

    # ---- ufn4.fl.run_federated, unrolled (same calls, same order, same seeds)
    run_seed = (derive_seed(seed, "poison", attack, rule) if poisoned
                else derive_seed(seed, "fed", method, alpha))
    mu = fcfg["fedprox_mu"] if method == "fedprox" else 0.0
    make_model = lambda: new_mlp(cfg1)                       # noqa: E731
    job.step(f"Train: {fcfg['rounds']} rounds x {len(agencies)} agencies, aggregation = {rule}")
    torch.manual_seed(run_seed)
    global_model = make_model()
    history = []
    for rnd in range(fcfg["rounds"]):
        g_params = get_params(global_model)
        results = []
        for a, (X, y) in enumerate(clients):
            local = make_model()
            set_params(local, g_params)
            local_train(local, X, y, fcfg["local_epochs"], cfg1["mlp"],
                        derive_seed(run_seed, "round", rnd, "agency", a), mu, g_params)
            params = get_params(local)
            if attacker is not None and a == attacker and attack == "scaled_update":
                f = pc["scaled_update"]["factor"]
                params = [g + f * (p - g) for g, p in zip(g_params, params)]
            results.append((params, len(y)))
        set_params(global_model, combine(results, rule, pc))
        pred = mlp_predict_proba(global_model, va["X"]).argmax(1)
        history.append(float(f1_score(va["y"], pred, average="macro")))
        job.progress = (rnd + 1) / fcfg["rounds"] * (0.8 if score else 1.0)
        job.log(f"round {rnd + 1:2d}: validation macro-F1 {history[-1]:.4f}")
    global_model.eval()
    job.done_step(f"final validation macro-F1 {history[-1]:.3f}")

    job.step("Compare with the model N4 stored for this setting")
    stored = (ROOT / cfg4["paths"]["models"] /
              (f"poison_{attack}_{rule}_s{seed}.pt" if poisoned else f"fed_{method}_a{alpha}_s{seed}.pt"))
    same = None
    if stored.exists():
        ref = load_model(cfg1, stored)
        same = all(np.array_equal(p, q) for p, q in zip(get_params(global_model), get_params(ref)))
    job.done_step({True: f"IDENTICAL to {stored.name}", False: f"differs from {stored.name}",
                   None: "N4 stored no model for this setting"}[same])

    out = {"alpha": alpha, "seed": seed, "method": method, "attack": attack, "rule": rule,
           "attacker": attacker, "class_counts": counts, "history": history,
           "stored_model": stored.name if stored.exists() else None, "identical_to_stored": same}
    if score:
        job.step("Score on N1's test cases: macro-F1 and TER (Integrated Gradients)")
        s, _ = score_model(global_model, te, keys["test"], ["integrated_gradients"], cfg1["explainers"],
                           cfg1["seed"], cfg1["oracle"]["min_prob_drop"],
                           cfg4["scoring"]["good_explanation_recall"])
        out["test"] = {k: v for k, v in s.items() if isinstance(v, float)}
        job.done_step(f"macro-F1 {s['macro_f1']:.3f}, TER {s['ter_integrated_gradients']:.3f}")
    return out
