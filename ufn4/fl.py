"""Cross-silo federated learning, simulated in one process.

Each round:
  1. the server sends the current global model to every agency;
  2. every agency trains it on its OWN cases for `local_epochs`
     (FedProx adds a penalty for drifting away from the global model);
  3. a malicious agency may tamper with its update (poisoning experiments);
  4. the server combines the returned models with an aggregation rule taken
     directly from Flower (flwr.server.strategy.aggregate):
        fedavg        - average weighted by number of cases     (McMahan et al., 2017)
        fedmedian     - element-wise median                     (Yin et al., 2018)
        fedtrimmedavg - mean after cutting the extremes         (Yin et al., 2018)
        krum          - keep the one model closest to the rest  (Blanchard et al., 2017)
Only model parameters ever leave an agency; raw cases never do.
Running everything in one process (no network) gives identical maths to a
Flower simulation and makes every run exactly repeatable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from flwr.server.strategy.aggregate import (aggregate, aggregate_krum, aggregate_median,
                                            aggregate_trimmed_avg)
from sklearn.metrics import f1_score

from ufn1.common import derive_seed
from ufn1.models import mlp_predict_proba


def get_params(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_params(model: nn.Module, params: list[np.ndarray]) -> None:
    sd = model.state_dict()
    model.load_state_dict({k: torch.tensor(p) for k, p in zip(sd.keys(), params)})


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, cfg_mlp: dict,
                seed: int, mu: float = 0.0, global_params: list[np.ndarray] | None = None) -> None:
    """Train `model` in place on one agency's cases. mu > 0 -> FedProx term."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_mlp["lr"], weight_decay=cfg_mlp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    anchor = [torch.tensor(p) for p in global_params] if mu > 0 else None
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), cfg_mlp["batch_size"]):
            idx = perm[i:i + cfg_mlp["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            if anchor is not None:                       # FedProx: mu/2 * ||w - w_global||^2
                loss = loss + mu / 2 * sum(((w - a) ** 2).sum()
                                           for w, a in zip(model.parameters(), anchor))
            loss.backward()
            opt.step()
    model.eval()


def combine(results: list[tuple[list[np.ndarray], int]], rule: str, pcfg: dict | None = None):
    """Aggregate client models with Flower's own implementation of each rule."""
    if rule in ("fedavg", "fedprox"):
        return aggregate(results)
    if rule == "fedmedian":
        return aggregate_median(results)
    if rule == "fedtrimmedavg":
        return aggregate_trimmed_avg(results, proportiontocut=pcfg["trimmed_proportion"])
    if rule == "krum":
        return aggregate_krum(results, num_malicious=pcfg["krum_num_malicious"], to_keep=0)
    raise ValueError(rule)


def run_federated(make_model, clients: list[tuple[np.ndarray, np.ndarray]], cfg_mlp: dict,
                  fcfg: dict, seed: int, method: str = "fedavg", rule: str | None = None,
                  attacker: int | None = None, attack: str = "none", attack_cfg: dict | None = None,
                  pcfg: dict | None = None, val: tuple[np.ndarray, np.ndarray] | None = None):
    """Return (global_model, history). history = validation macro-F1 after each round,
    recorded for the convergence plot only (never used to pick the model)."""
    rule = rule or method
    mu = fcfg["fedprox_mu"] if method == "fedprox" else 0.0
    torch.manual_seed(seed)
    global_model = make_model()
    history = []
    for rnd in range(fcfg["rounds"]):
        g_params = get_params(global_model)
        results = []
        for a, (X, y) in enumerate(clients):
            local = make_model()
            set_params(local, g_params)
            local_train(local, X, y, fcfg["local_epochs"], cfg_mlp,
                        derive_seed(seed, "round", rnd, "agency", a), mu, g_params)
            params = get_params(local)
            if attacker is not None and a == attacker and attack == "scaled_update":
                f = attack_cfg["scaled_update"]["factor"]          # send g + f * (real update)
                params = [g + f * (p - g) for g, p in zip(g_params, params)]
            results.append((params, len(y)))
        set_params(global_model, combine(results, rule, pcfg))
        if val is not None:
            pred = mlp_predict_proba(global_model, val[0]).argmax(1)
            history.append(float(f1_score(val[1], pred, average="macro")))
    global_model.eval()
    return global_model, history
