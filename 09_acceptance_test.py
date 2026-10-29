"""STEP 11 - The explanation acceptance test (the XAI equivalent of tool validation).

Level 1 - METHOD validation (uses the benchmark's ground truth; done once per explainer):
    (i)  mean strict recall@|G| >= method_min_recall on EVERY tier, and
    (ii) its F-Fidelity FFid+ 95% CI lies entirely above the random explainer's.
Level 2 - CASE check (uses NO ground truth -> usable on real casework):
    an explanation is printed as 'VALIDATED' only if
    (a) the explainer passed Level 1,
    (b) primary and secondary explainers agree: top-5 feature agreement >= case_min_agreement
        (guards against manipulated / unstable explanations - see step 12), and
    (c) the model's confidence in its prediction >= tau.
    Otherwise the report prints 'NOT VALIDATED: <reasons>' instead of stating it as fact.
    tau is NOT chosen by hand: it is the smallest value on the confidence grid for which,
    on the VALIDATION split, at least target_precision of validated explanations are
    correct. That tau is then applied unchanged to the TEST split.
Finally we use the hidden answer key to measure how well Level 2 works on test.

Output: results/acceptance_cases.csv, results/acceptance_summary.json
Run:  python 09_acceptance_test.py
"""
import numpy as np
import pandas as pd
import torch

from ufn1.common import banner, get_path, load_config, load_json, save_json, sha256_of
from ufn1.explainers import EXPLAINERS
from ufn1.metrics import feature_agreement, recall_at_k
from ufn1.models import TriageMLP, mlp_predict_proba
from ufn1.schema import CLASSES, FEATURES

cfg = load_config()
acc = cfg["acceptance"]
data_dir, model_dir, res_dir = (get_path(cfg, k) for k in ("data", "models", "results"))

# ---- refuse to run if thresholds were edited after they were frozen -------
frozen = load_json(res_dir / "frozen_thresholds.json")
if frozen["sha256"] != sha256_of(acc):
    raise SystemExit("Acceptance settings in config.yaml differ from the copy frozen in step 3.\n"
                     "Restore them, or re-run everything from step 3 and REPORT the change.")
print(f"Acceptance settings verified against the frozen copy ({frozen['frozen_at']}).")

K, tiers = cfg["scoring"]["k_fixed"], cfg["benchmark"]["tiers"]
P, S = acc["primary_explainer"], acc["secondary_explainer"]
mlp = TriageMLP(len(FEATURES), len(CLASSES), cfg["mlp"]["hidden"], cfg["mlp"]["dropout"])
mlp.load_state_dict(torch.load(model_dir / "mlp.pt", weights_only=True))
mlp.eval()

# ------------------------------------------------------------------ Level 1
banner(f"Level 1 - method validation (recall@|G| >= {acc['method_min_recall']} on every tier,"
       " FFid+ above random)")
gt = pd.read_csv(res_dir / "table_groundtruth.csv")
ff = pd.read_csv(res_dir / "table_ffidelity.csv")
ff_all = ff[ff.tier == "ALL"].set_index("explainer")
rand_hi = ff_all.at["random", "ffid_plus_hi"]
level1 = {}
print(f"  {'explainer':22s}" + "".join(f"{t:>9s}" for t in tiers) + f"{'FFid+ lo':>10s}   verdict")
for name in EXPLAINERS:
    r = gt[gt.explainer == name].set_index("tier")
    rec_ok = {t: bool(r.at[t, "recall@|G|"] >= acc["method_min_recall"]) for t in tiers}
    ffid_ok = bool(ff_all.at[name, "ffid_plus_lo"] > rand_hi) if acc["method_ffid_beats_random"] else True
    ok = all(rec_ok.values()) and ffid_ok
    level1[name] = {"recall_by_tier": {t: float(r.at[t, "recall@|G|"]) for t in tiers},
                    "recall_ok_by_tier": rec_ok, "ffid_beats_random": ffid_ok, "validated": ok}
    print(f"  {name:22s}" + "".join(f"{r.at[t, 'recall@|G|']:9.3f}" for t in tiers)
          + f"{ff_all.at[name, 'ffid_plus_lo']:10.3f}   " + ("VALIDATED" if ok else "not validated"))
print(f"  (random explainer FFid+ CI upper bound = {rand_hi:.3f})")
if not level1[P]["validated"]:
    print(f"\n  NOTE: primary explainer '{P}' failed Level 1 -> every case will be NOT VALIDATED.")


# ------------------------------------------------------------------ Level 2
def case_table(split: str, att_file: str) -> pd.DataFrame:
    """Per offence case: agreement, confidence, and (hidden) correctness."""
    A = np.load(res_dir / att_file)
    keys = load_json(data_dir / f"{split}_answer_key.json")
    X = np.load(data_dir / f"{split}.npz")["X"][A["case_index"]]
    proba = mlp_predict_proba(mlp, X)
    rows = []
    for j, i in enumerate(A["case_index"]):
        k = keys[int(i)]
        G = set(k["primary_idx"])
        rows.append({"case_index": int(i), "case_id": k["case_id"], "label": k["label"], "tier": k["tier"],
                     "predicted": CLASSES[int(A["target"][j])],
                     "agreement_FA5": feature_agreement(A[P][j], A[S][j], K),
                     "confidence": float(proba[j].max()),
                     "recall@|G|": recall_at_k(A[P][j], G, len(G)) if
                                   CLASSES[int(A["target"][j])] == k["label"] else 0.0})
    d = pd.DataFrame(rows)
    d["good_explanation"] = d["recall@|G|"] >= acc["good_explanation_recall"]
    return d


def validated(d: pd.DataFrame, tau: float) -> pd.Series:
    return (level1[P]["validated"] & (d.agreement_FA5 >= acc["case_min_agreement"])
            & (d.confidence >= tau))


banner(f"Level 2 - calibrate the confidence cut-off on the VALIDATION split "
       f"(target precision {acc['target_precision']})")
val = case_table("val", "attributions_val.npz")
tau, calib = None, []
for t in acc["confidence_grid"]:
    v = validated(val, t)
    prec = float(val.good_explanation[v].mean()) if v.any() else float("nan")
    calib.append({"tau": t, "validated_rate": float(v.mean()), "precision": prec})
    print(f"  tau={t:<6} validated={v.mean():.3f}  precision={prec:.3f}")
    if tau is None and v.any() and prec >= acc["target_precision"]:
        tau = t
if tau is None:
    tau = acc["confidence_grid"][-1]
    print("  target precision not reached on validation -> using the strictest cut-off")
print(f"  -> calibrated tau = {tau}  (now FROZEN for the test split)")

banner("Level 2 - apply to the TEST split and score it with the hidden answer key")
test = case_table("test", "attributions_mlp.npz")
oracle = {o["case_index"]: o["passes"] for o in load_json(res_dir / "oracle_cases.json") if o["model"] == "mlp"}
test["oracle_pass"] = test.case_index.map(oracle)
test["validated"] = validated(test, tau)


def reasons(r):
    out = []
    if not level1[P]["validated"]:
        out.append(f"{P} failed method validation")
    if r.agreement_FA5 < acc["case_min_agreement"]:
        out.append(f"explainers disagree (top-{K} agreement {r.agreement_FA5:.2f})")
    if r.confidence < tau:
        out.append(f"model not confident ({r.confidence:.3f} < {tau})")
    return "VALIDATED" if not out else "NOT VALIDATED: " + "; ".join(out)


test["report_label"] = test.apply(reasons, axis=1)
test.to_csv(res_dir / "acceptance_cases.csv", index=False)


def summarise(d):
    v, g = d.validated, d.good_explanation
    return {"n": int(len(d)), "share_good": float(g.mean()), "validated_rate": float(v.mean()),
            "precision_of_validated": float(g[v].mean()) if v.any() else None,
            "bad_caught_rate": float((~v[~g]).mean()) if (~g).any() else None,
            "good_rejected_rate": float((~v[g]).mean()) if g.any() else None}


summary = {"settings": acc, "level1": level1, "calibration_val": calib, "tau": tau, "level2_test": {}}
f3 = lambda x: "   n/a" if x is None else f"{x:6.3f}"
print(f"  {'subset':18s} {'n':>4s} {'good':>6s} {'validated':>10s} {'P(good|valid)':>14s}"
      f" {'bad caught':>11s} {'good rejected':>14s}")
for label, d in ([("all offence cases", test), ("oracle-passed", test[test.oracle_pass]),
                  ("oracle-failed", test[~test.oracle_pass])] + [(f"tier={t}", test[test.tier == t]) for t in tiers]):
    if len(d):
        sm = summarise(d)
        summary["level2_test"][label] = sm
        print(f"  {label:18s} {sm['n']:4d} {sm['share_good']:6.3f} {sm['validated_rate']:10.3f} "
              f"{f3(sm['precision_of_validated']):>14s} {f3(sm['bad_caught_rate']):>11s} {f3(sm['good_rejected_rate']):>14s}")
save_json(summary, res_dir / "acceptance_summary.json")
print("\n  'good' = the explanation really found the planted evidence (hidden answer key).")
print("  Without the test, every explanation is shown as fact: precision = the 'good' column.")

print("\n  Example report lines:")
for _, r in test.groupby("validated").head(2).iterrows():
    print(f"    {r.case_id} ({r.label}, {r.tier}, predicted {r.predicted}): {r.report_label}")
print("\nDone. Optional: python 10_scaffold_attack.py   |   Then: python 11_make_report.py")
