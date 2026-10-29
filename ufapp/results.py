"""Read the results N1-N5 wrote and hand them to the web page as compact JSON.

Nothing is typed in: every number is computed here from the CSV/JSON files of the
run on THIS machine, so the dashboard always shows your own run. Means are over
the seeds the novelty used; 'sd' is the standard deviation over those seeds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ufn1.common import ROOT, load_json

from .store import MissingOutput, need


def _clean(x):
    """NaN/inf -> None, numpy -> python (JSON-safe)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.to_dict(orient="records"))


def _try(fn):
    try:
        return {"ok": True, **fn()}
    except MissingOutput as e:
        return {"ok": False, "missing": str(e)}


# ---------------------------------------------------------------- N1
def n1() -> dict:
    def f():
        r = ROOT / "results"
        perf = load_json(need(r / "model_performance.json", "N1 step 6"))
        gt = pd.read_csv(need(r / "table_groundtruth.csv", "N1 step 9"))
        acc = load_json(need(r / "acceptance_summary.json", "N1 step 11"))
        ffid = pd.read_csv(need(r / "table_ffidelity.csv", "N1 step 10"))
        oracle = load_json(need(r / "oracle_global.json", "N1 step 7"))
        scaffold = load_json(r / "scaffold_attack.json") if (r / "scaffold_attack.json").exists() else None
        return {"models": {k: v for k, v in perf.items() if k in ("mlp", "xgb")},
                "recall_by_tier": _records(gt[["explainer", "tier", "n_cases", "recall@|G|", "recall@|G|_lo",
                                               "recall@|G|_hi", "precision@5", "top1_hit"]]),
                "ffidelity": _records(ffid[["explainer", "tier", "ffid_plus", "ffid_plus_lo", "ffid_plus_hi"]]),
                "acceptance": {"tau": acc["tau"], "settings": acc["settings"],
                               "level1": {k: {"validated": v["validated"], "recall_by_tier": v["recall_by_tier"]}
                                          for k, v in acc["level1"].items()},
                               "level2_test": acc["level2_test"]},
                "oracle": oracle, "scaffold": scaffold,
                "figures": ["fig1_recall_by_tier.png", "fig2_disagreement_FA.png", "fig3_ffidelity.png"]}
    return _try(f)


# ---------------------------------------------------------------- N2
def n2() -> dict:
    def f():
        r = ROOT / "results_n2"
        ra = pd.read_csv(need(r / "release_attack.csv", "N2 step 5"))
        g = (ra.groupby(["scheme", "eps"]).agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                                               tpr_1pct=("tpr_at_1pct_fpr", "mean"),
                                               evidence_recall3=("evidence_recall3", "mean"))
             .reset_index().sort_values(["scheme", "eps"]))
        game = pd.read_csv(need(r / "game.csv", "N2 step 6"))
        gg = game.groupby(["attack", "m", "eps"]).auc.mean().reset_index()
        mia = pd.read_csv(need(r / "case_mia.csv", "N2 step 8"))
        mm = mia.groupby(["target", "statistic", "test"]).auc.mean().reset_index()
        return {"release": _records(g), "game": _records(gg), "case_mia": _records(mm),
                "figures": ["fig9_privacy_vs_faithfulness.png", "fig10_group_size.png", "fig11_case_mia.png"]}
    return _try(f)


# ---------------------------------------------------------------- N3 (teammate)
def n3() -> dict:
    def f():
        r = ROOT / "results_n5"
        eq = load_json(need(r / "n3_equivalence.json", "N5 step 3 (n5_01)"))
        on1 = load_json(need(r / "n3_on_n1_databases.json", "N5 step 5 (n5_03)"))
        rel = pd.read_csv(need(r / "reliability.csv", "N5 step 5 (n5_03)"))
        return {"equivalence": {"identical": eq["identical"], "numpy_seed": eq["numpy_seed"],
                                "platt_a": eq["ours"]["platt_a"], "platt_b": eq["ours"]["platt_b"],
                                "ece_in_sample": eq["teammate_run_pipeline"]["ece_in_sample"]},
                "teammate_benchmark": eq["teammate_benchmark_summary"],
                "teammate_pragma": eq["teammate_pragma_sensitivity"],
                "on_n1_databases": on1, "reliability": _records(rel)}
    return _try(f)


# ---------------------------------------------------------------- N4
def n4() -> dict:
    def f():
        r = ROOT / "results_n4"
        s = pd.read_csv(need(r / "model_scores.csv", "N4 step 8"),
                        dtype={"method": str, "attack": str, "rule": str})
        cols = ["macro_f1", "ter_integrated_gradients", "recall_integrated_gradients", "oracle_pass_rate"]
        central = s[s.kind == "central"][cols].agg(["mean", "std"]).T.reset_index()
        fed = (s[s.kind == "federated"].groupby(["method", "alpha"])[cols].agg(["mean", "std"]))
        fed.columns = [f"{a}_{b}" for a, b in fed.columns]
        local = s[s.kind == "local"].groupby(["alpha", "seed"])[cols].mean().groupby("alpha").agg(["mean", "std"])
        local.columns = [f"{a}_{b}" for a, b in local.columns]
        pois = (s[s.kind == "poison"].groupby(["attack", "rule"])[["macro_f1", "f1_fraud", "ter_integrated_gradients"]]
                .mean().reset_index())
        conv = load_json(need(r / "convergence.json", "N4 step 6"))
        conv_mean = {}
        for k, hist in conv.items():
            m, a, _ = k.split("|")
            conv_mean.setdefault(f"{m}|{a}", []).append(hist)
        conv_mean = {k: np.mean(v, axis=0).round(4).tolist() for k, v in conv_mean.items()}
        parts = pd.read_csv(need(r / "partition_counts.csv", "N4 step 4"))
        case = load_json(need(r / "case_export_check.json", "N4 step 3"))
        return {"central": _records(central), "federated": _records(fed.reset_index()),
                "local": _records(local.reset_index()), "poisoning": _records(pois),
                "convergence": conv_mean, "partitions": _records(parts), "case_export": case,
                "figures": ["fig4_accuracy_and_ter_vs_alpha.png", "fig5_class_rarity.png", "fig6_poisoning.png",
                            "fig7_proxy_share.png", "fig8_convergence.png"]}
    return _try(f)


# ---------------------------------------------------------------- N5
def n5() -> dict:
    def f():
        r = ROOT / "results_n5"
        g = pd.read_csv(need(r / "eval_grid.csv", "N5 step 8"), dtype={"rho": str, "level": str, "foreign_rate": str})
        grid = g[(g.foreign_rate == "0.0") & (g.level != "benign_all")]
        f1 = (grid.groupby(["variant", "rho", "level"]).macro_f1.agg(["mean", "std"]).reset_index())
        f1["level"] = f1.level.astype(float)
        ben = (g[g.level == "benign_all"].groupby(["variant", "rho", "foreign_rate"]).false_incrimination
               .mean().reset_index())
        ex = pd.read_csv(need(r / "eval_explain.csv", "N5 step 8"), dtype={"rho": str})
        ter = ex.groupby(["variant", "rho", "level"])[["ter", "evidence_recall"]].mean().reset_index()
        fl = pd.read_csv(need(r / "flag.csv", "N5 step 9"), dtype={"rho": str, "level": str})
        fl_sum = fl.groupby(["variant", "rho", "group", "level"])[["flag_rate", "share_of_fi", "n"]].mean().reset_index()
        cf = pd.read_csv(need(r / "conformal.csv", "N5 step 10"), dtype={"rho": str, "level": str})
        cf_sum = (cf.groupby(["variant", "rho", "level", "foreign_rate"])
                  [["coverage_full", "size_full", "coverage_live", "size_live"]].mean().reset_index())
        fed = pd.read_csv(need(r / "federated.csv", "N5 step 11"), dtype={"level": str})
        dp = pd.read_csv(need(r / "dp_release.csv", "N5 step 13"))
        dps = dp.groupby("eps")[["abs_error", "evidence_recall_at_3"]].mean().reset_index()
        on1 = load_json(need(r / "n3_on_n1_databases.json", "N5 step 5"))
        case = load_json(need(r / "case_export_check.json", "N5 step 12"))
        return {"macro_f1": _records(f1), "false_incrimination": _records(ben), "ter": _records(ter),
                "flag": _records(fl_sum), "conformal": _records(cf_sum), "federated": _records(fed),
                "dp": _records(dps), "calibration": on1["calibration"], "case_export": case,
                "figures": ["fig15_recovery.png", "fig16_calibration.png", "fig17_triage_vs_deletion.png",
                            "fig18_false_incrimination.png", "fig19_flag.png", "fig20_conformal.png"]}
    return _try(f)
