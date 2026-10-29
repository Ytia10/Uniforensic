"""N2 STEP 8 - Do PER-CASE explanations reveal which cases a model trained on?

Setting: an investigator or opposing party can query the model and see the
explanation of any case. Can they tell whether that case was in training?
Targets (all explaining the predicted class with Integrated Gradients):
  overfit_control - positive control (memorises; attacks MUST work here)
  centralised     - all data pooled (N4 baseline, = N1's model for seed 2026)
  federated       - the deployed FedAvg model (N4)
  local_only      - the 8 agencies' own models (N4), pooled
Members = cases the target trained on; non-members = N1 test cases with the SAME
crime-type mix as the members (class-matched). Without matching, an agency model
would be "caught" simply because it explains crime types it never saw differently.
Statistics of the explanation vector: variance (Shokri et al.), L1 and L2 norm
(Huang et al.). Two tests per statistic:
  threshold - one global threshold; its direction is learned from shadow models
  lrt       - per-case calibration: the shadow models never saw the target case,
              so they show what its explanation looks like when it is NOT a member;
              score = how far the target model's value is from that (z-score).
              (offline variant of the likelihood-ratio idea of Huang et al. / LiRA)
Output: results_n2/case_mia.csv
Run:  python n2_06_case_mia.py      (about 2-4 minutes)
"""
import time

import numpy as np
import pandas as pd

from ufn1.common import banner, derive_seed, load_json
from ufn2.attacks import auc, auc_ci, explanation_stats, tpr_at_fpr
from ufn2.common import deployed_model, explain, load_config_n2, load_n1, load_n4, path_n2
from ufn4.common import load_model

STATS = ("variance", "l1", "l2")
cfg2 = load_config_n2()
mc = cfg2["case_mia"]
cfg1, data, keys = load_n1()
cfg4, parts, mdir4 = load_n4()
seed = cfg4["run_seeds"][cfg2["analysis_seed_index"]]
alpha = cfg2["deployed"]["alpha"]
mdir2 = path_n2(cfg2, "models")
reg = load_json(mdir2 / "registry.json")
sh = np.load(path_n2(cfg2, "data") / "shadow.npz")
tr, te = data["train"]["X"], data["test"]["X"]
rng = np.random.default_rng(derive_seed(seed, "case_mia"))


def stats_of(model, X):
    attr, _ = explain(model, X, cfg1, cfg2, cfg1["seed"])
    return explanation_stats(attr)


y_tr, y_te = data["train"]["y"], data["test"]["y"]


def matched_nonmembers(members: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """One test case of the same true class for every member (no repeats per target)."""
    out = []
    for c in np.unique(y_tr[members]):
        need = int((y_tr[members] == c).sum())
        pool = np.where(y_te == c)[0]
        out.append(r.choice(pool, size=need, replace=need > len(pool)))
    return np.concatenate(out)


def local_training_cases(a: int, idx: list[int]) -> np.ndarray:
    """Agency a's training cases, split exactly as in N4 step 5 (holdout removed)."""
    idx = np.array(idx)
    np.random.default_rng(derive_seed(seed, "holdout", alpha, a)).shuffle(idx)
    n_ho = max(1, int(round(cfg4["local_only"]["holdout"] * len(idx))))
    return idx[n_ho:]


# ------------------------------------------------ targets: (model, members, non-members)
ctrl = next(r for r in reg if r["kind"] == "overfit_control")
ctrl_mem = np.array(ctrl["members"])
mem_big = rng.choice(len(tr), size=mc["candidates"], replace=False)
non_big = matched_nonmembers(mem_big, rng)
per = mc["candidates"] // cfg4["agencies"]
local_parts = []
for a, idx in enumerate(parts[f"{alpha}|{seed}"]):
    own = local_training_cases(a, idx)
    mem = rng.choice(own, size=min(per, len(own)), replace=False)
    local_parts.append((load_model(cfg1, mdir4 / f"local_a{alpha}_s{seed}_ag{a}.pt"),
                        mem, matched_nonmembers(mem, rng)))
targets = {
    "overfit_control": ("overfit", [(load_model(cfg1, mdir2 / ctrl["file"]), ctrl_mem,
                                     matched_nonmembers(ctrl_mem, rng))]),
    "centralised": ("standard", [(load_model(cfg1, mdir4 / f"central_s{seed}.pt"), mem_big, non_big)]),
    "federated": ("standard", [(deployed_model(cfg1, cfg2, cfg4, seed), mem_big, non_big)]),
    "local_only": ("small", local_parts),
}

# ------------------------------------------------ attack every target
rows, t0 = [], time.time()
for name, (recipe, tparts) in targets.items():
    shadows = [r for r in reg if r["kind"] == "shadow" and r["recipe"] == recipe]
    shadow_models = [load_model(cfg1, mdir2 / r["file"]) for r in shadows]
    banner(f"Target '{name}': {sum(len(m) for _, m, _ in tparts)} members vs "
           f"{sum(len(n) for _, _, n in tparts)} non-members; {len(shadows)} '{recipe}' shadow models")
    # 1. direction of each statistic, learned ONLY from the attacker's shadow models
    diffs = {k: [] for k in STATS}
    for r, sm in zip(shadows, shadow_models):
        mem = np.array(r["members"])
        out = np.setdiff1d(np.arange(len(sh["y"])), mem)
        pick = np.random.default_rng(derive_seed(seed, "direction", r["file"]))
        s_m = stats_of(sm, sh["X"][pick.choice(mem, 200, replace=False)])
        s_o = stats_of(sm, sh["X"][pick.choice(out, 200, replace=False)])
        for k in STATS:
            diffs[k].append(s_m[k].mean() - s_o[k].mean())
    direction = {k: 1.0 if np.mean(diffs[k]) > 0 else -1.0 for k in STATS}
    # 2. target scores, and per-case "non-member" reference from the shadow models
    score = {(g, t, k): [] for g in ("in", "out") for t in ("threshold", "lrt") for k in STATS}
    for model, mem, non in tparts:
        for group, X in (("in", tr[mem]), ("out", te[non])):
            st = stats_of(model, X)
            ref = [stats_of(sm, X) for sm in shadow_models]
            for k in STATS:
                R = np.stack([r_[k] for r_ in ref])                  # shadows x cases
                score[(group, "threshold", k)].append(direction[k] * st[k])
                score[(group, "lrt", k)].append(direction[k] * (st[k] - R.mean(0)) / (R.std() + 1e-12))
    for k in STATS:
        for test in ("threshold", "lrt"):
            si = np.concatenate(score[("in", test, k)])
            so = np.concatenate(score[("out", test, k)])
            lo, hi = auc_ci(si, so, 1000, derive_seed(seed, "ci", name, k, test))
            rows.append({"seed": seed, "target": name, "statistic": k, "test": test,
                         "members_score_higher": direction[k] > 0, "auc": auc(si, so),
                         "auc_lo": lo, "auc_hi": hi,
                         "tpr_at_1pct_fpr": tpr_at_fpr(si, so, mc["tpr_at_fpr"]),
                         "n_members": len(si), "n_nonmembers": len(so)})
            print(f"  {k:8s} {test:9s} AUC {rows[-1]['auc']:.3f} [{lo:.3f}, {hi:.3f}]"
                  f"   TPR@1%FPR {rows[-1]['tpr_at_1pct_fpr']:.3f}")
    print(f"  [{time.time() - t0:.0f}s]")
pd.DataFrame(rows).to_csv(path_n2(cfg2, "results") / "case_mia.csv", index=False)
print("\nDone. Next: python n2_07_make_report.py")
