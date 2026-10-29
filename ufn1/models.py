"""Triage models: an MLP (explained by Captum/SHAP) and an XGBoost reference
(explained by exact TreeSHAP)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn


class TriageMLP(nn.Module):
    """Outputs LOGITS. Use ProbWrapper when an explainer needs probabilities."""

    def __init__(self, n_in: int, n_out: int, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProbWrapper(nn.Module):
    """Model -> softmax probabilities. Every explainer explains P(class | case),
    so attributions from all methods are on the same scale."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def train_mlp(Xtr, ytr, Xva, yva, cfg_mlp: dict, n_classes: int, seed: int,
              mask_beta: float = 0.0, init_model: nn.Module | None = None,
              max_epochs: int | None = None, verbose: bool = True) -> nn.Module:
    """Train with early stopping on validation loss.
    mask_beta > 0 -> F-Fidelity fine-tuning: each training row gets a random
    fraction beta of its features set to 0 (the training mean)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = init_model if init_model is not None else TriageMLP(
        Xtr.shape[1], n_classes, cfg_mlp["hidden"], cfg_mlp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg_mlp["lr"], weight_decay=cfg_mlp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva, dtype=torch.long)
    d = Xtr.shape[1]
    n_mask = int(np.floor(mask_beta * d))

    best, best_state, bad = np.inf, None, 0
    epochs = max_epochs or cfg_mlp["max_epochs"]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), cfg_mlp["batch_size"]):
            idx = perm[i:i + cfg_mlp["batch_size"]]
            xb = Xtr_t[idx].clone()
            if n_mask > 0:                      # random masking (F-Fidelity, Eq. for L)
                scores = torch.rand(xb.shape, generator=g)
                cols = scores.argsort(dim=1)[:, :n_mask]
                xb.scatter_(1, cols, 0.0)
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), yva_t).item()
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0):
            print(f"  epoch {ep:3d}  val_loss={vl:.4f}")
        if bad >= cfg_mlp["patience"]:
            if verbose:
                print(f"  early stop at epoch {ep}  (best val_loss={best:.4f})")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()


def train_xgb(Xtr, ytr, cfg_xgb: dict, seed: int):
    from xgboost import XGBClassifier
    clf = XGBClassifier(eval_metric="mlogloss",   # objective is chosen from the labels
                        random_state=seed, n_jobs=1, tree_method="hist", **cfg_xgb)
    clf.fit(Xtr, ytr)
    return clf


def xgb_treeshap(clf, X: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP computed natively by XGBoost (pred_contribs=True).
    Returns array (n_cases, n_classes, n_features) in log-odds (margin) units."""
    from xgboost import DMatrix
    contrib = clf.get_booster().predict(DMatrix(X), pred_contribs=True)
    # multiclass: (n, n_classes, d+1); binary: (n, d+1). Last column = bias term.
    if contrib.ndim == 2:
        contrib = contrib[:, None, :]
    return contrib[:, :, :-1]
