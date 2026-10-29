"""Run several explainers on the SAME model and the SAME cases.

All explainers explain P(target class | case) of the MLP, with 'feature removed'
meaning 'set to 0' = the training mean in the scaled space.

    kernelshap            - shap.KernelExplainer (sampling, model-agnostic)
    integrated_gradients  - Captum (gradient path from baseline 0)
    gradientshap          - Captum (expected gradients over a background sample)
    lime                  - Captum (local linear surrogate)
    feature_ablation      - Captum (remove one feature at a time)
    random                - random numbers: the floor every real method must beat
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch

from .common import derive_seed
from .models import ProbWrapper

EXPLAINERS = ["kernelshap", "integrated_gradients", "gradientshap", "lime",
              "feature_ablation", "random"]


def _pick_target(sv, targets):
    """shap returns either a list (one array per class) or (n, d, C) array."""
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    return sv[np.arange(len(targets)), :, targets]


def run_explainer(name: str, model: torch.nn.Module, X: np.ndarray, targets: np.ndarray,
                  background: np.ndarray, ecfg: dict, seed: int, progress: bool = True) -> np.ndarray:
    """Return attributions of shape (n_cases, n_features)."""
    pm = ProbWrapper(model).eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(targets, dtype=torch.long)
    zeros = torch.zeros_like(Xt)
    t0 = time.time()

    if name == "kernelshap":
        import shap

        def f(z):
            with torch.no_grad():
                return pm(torch.tensor(z, dtype=torch.float32)).numpy()
        expl = shap.KernelExplainer(f, background)
        out = np.zeros_like(X)
        for i in range(len(X)):
            np.random.seed(derive_seed(seed, "kernelshap", i))     # reproducible per case
            sv = expl.shap_values(X[i:i + 1], nsamples=ecfg["nsamples"], silent=True)
            out[i] = _pick_target(sv, targets[i:i + 1])[0]
            if progress and i % 50 == 0:
                print(f"    kernelshap {i}/{len(X)}")
        attr = out

    elif name == "integrated_gradients":
        from captum.attr import IntegratedGradients
        attr = IntegratedGradients(pm).attribute(Xt, baselines=zeros, target=tt,
                                                 n_steps=ecfg["n_steps"]).detach().numpy()

    elif name == "gradientshap":
        from captum.attr import GradientShap
        torch.manual_seed(derive_seed(seed, "gradientshap"))
        attr = GradientShap(pm).attribute(Xt, baselines=torch.tensor(background, dtype=torch.float32),
                                          target=tt, n_samples=ecfg["n_samples"],
                                          stdevs=0.0).detach().numpy()

    elif name == "lime":
        from captum.attr import Lime
        from captum._utils.models.linear_model import SkLearnLinearRegression
        lime = Lime(pm, interpretable_model=SkLearnLinearRegression())
        out = np.zeros_like(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(X)):
                torch.manual_seed(derive_seed(seed, "lime", i))
                out[i] = lime.attribute(Xt[i:i + 1], target=int(targets[i]), baselines=zeros[i:i + 1],
                                        n_samples=ecfg["n_samples"]).detach().numpy()[0]
                if progress and i % 100 == 0:
                    print(f"    lime {i}/{len(X)}")
        attr = out

    elif name == "feature_ablation":
        from captum.attr import FeatureAblation
        attr = FeatureAblation(pm).attribute(Xt, baselines=0.0, target=tt).detach().numpy()

    elif name == "random":
        attr = np.random.default_rng(derive_seed(seed, "random")).normal(size=X.shape)

    else:
        raise ValueError(name)

    if progress:
        print(f"  {name:22s} done in {time.time() - t0:6.1f}s")
    return attr.astype(np.float64)
