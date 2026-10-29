"""Trace every headline number the Workbench shows back to the result file it comes from.

For each number this script:
  1. reads it from the app, exactly as the web page receives it (/api/overview, /api/results/nX);
  2. recomputes it straight from the N1-N5 result file with pandas/json, WITHOUT the app's code;
  3. prints both, the file and the rule used, and whether they agree.

It also traces the most recent case analysis (app_data/runs/<newest>/) back to its own files.

Run:   python trace_results.py            (no server needed; takes a few seconds)
Writes app_data/trace_results.csv as a record. Changes nothing else.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

ROWS: list[dict] = []


def j(p: str):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def add(screen: str, what: str, shown, source: str, rule: str, recomputed, tol: float = 1e-9):
    if isinstance(shown, (int, float)) and isinstance(recomputed, (int, float)):
        ok = abs(float(shown) - float(recomputed)) <= tol
    else:
        ok = shown == recomputed
    ROWS.append({"screen": screen, "number": what, "app_shows": shown, "recomputed_from_file": recomputed,
                 "file": source, "rule": rule, "match": "yes" if ok else "NO"})


def app_values():
    from fastapi.testclient import TestClient
    from ufapp.main import app
    c = TestClient(app)
    out = {"overview": c.get("/api/overview").json()}
    for n in ("n1", "n2", "n3", "n4", "n5"):
        out[n] = c.get(f"/api/results/{n}").json()
    return out


def headline(ov, n):
    return next(h for h in ov["headlines"] if h["novelty"] == n)["value"]


def main() -> int:
    A = app_values()
    ov = A["overview"]

    # ------------------------------------------------------------ N1
    acc = j("results/acceptance_summary.json")
    perf = j("results/model_performance.json")
    l2 = acc["level2_test"]["all offence cases"]
    add("Overview, N1", "validated explanations that are right", headline(ov, "N1"),
        "results/acceptance_summary.json", 'level2_test["all offence cases"].precision_of_validated', l2["precision_of_validated"])
    add("N1", "acceptance threshold tau", A["n1"]["acceptance"]["tau"], "results/acceptance_summary.json", "tau", acc["tau"])
    add("N1", "bad explanations caught", A["n1"]["acceptance"]["level2_test"]["all offence cases"]["bad_caught_rate"],
        "results/acceptance_summary.json", 'level2_test["all offence cases"].bad_caught_rate', l2["bad_caught_rate"])
    add("N1", "triage macro-F1 (MLP)", A["n1"]["models"]["mlp"]["macro_f1"], "results/model_performance.json", "mlp.macro_f1",
        perf["mlp"]["macro_f1"])
    gt = pd.read_csv(ROOT / "results/table_groundtruth.csv")
    r = gt[(gt.explainer == "integrated_gradients") & (gt.tier == "subtle")]["recall@|G|"].iloc[0]
    shown = next(x for x in A["n1"]["recall_by_tier"] if x["explainer"] == "integrated_gradients" and x["tier"] == "subtle")["recall@|G|"]
    add("N1", "Level 1 recall, Integrated Gradients, subtle tier", shown, "results/table_groundtruth.csv",
        "row explainer=integrated_gradients, tier=subtle, column recall@|G|", r)

    # ------------------------------------------------------------ N2
    ra = pd.read_csv(ROOT / "results_n2/release_attack.csv")
    none = ra[ra.scheme == "none"].auc.mean()
    sdp1 = ra[(ra.scheme == "secagg_dp") & (ra.eps.astype(str).isin(["1", "1.0"]))].auc.mean()
    sa = ra[ra.scheme == "secagg"].auc.mean()
    add("Overview, N2", "attack AUC, secure aggregation + DP, epsilon 1", headline(ov, "N2"), "results_n2/release_attack.csv",
        "mean of auc over seeds and repeats where scheme=secagg_dp and eps=1", sdp1)
    pick = lambda s: next(x for x in A["n2"]["release"] if x["scheme"] == s)["auc"]  # noqa: E731
    add("N2", "attack AUC, no protection", pick("none"), "results_n2/release_attack.csv", "mean auc where scheme=none", none)
    add("N2", "attack AUC, secure aggregation", pick("secagg"), "results_n2/release_attack.csv", "mean auc where scheme=secagg", sa)

    # ------------------------------------------------------------ N3 (teammate; measured by N5)
    on1 = j("results_n5/n3_on_n1_databases.json")
    eq = j("results_n5/n3_equivalence.json")
    add("Overview, N3", "ECE after N5 recalibration", headline(ov, "N3"), "results_n5/n3_on_n1_databases.json",
        "calibration.ece_recalibrated", on1["calibration"]["ece_recalibrated"])
    add("N3", "ECE of N3 confidence as shipped", A["n3"]["on_n1_databases"]["calibration"]["ece_raw"],
        "results_n5/n3_on_n1_databases.json", "calibration.ece_raw", on1["calibration"]["ece_raw"])
    add("N3", "N3 recall on their benchmark", A["n3"]["teammate_benchmark"]["n3"]["recall"], "results_n5/n3_equivalence.json",
        "teammate_benchmark_summary.n3.recall", eq["teammate_benchmark_summary"]["n3"]["recall"])
    add("N3", "teammate pipeline reproduced bit for bit", A["n3"]["equivalence"]["identical"], "results_n5/n3_equivalence.json",
        "identical", eq["identical"])

    # ------------------------------------------------------------ N4
    s = pd.read_csv(ROOT / "results_n4/model_scores.csv", dtype={"method": str, "attack": str, "rule": str})
    central = s[s.kind == "central"].macro_f1.mean()
    fed = s[(s.kind == "federated") & (s.method == "fedavg") & (s.alpha == 0.5)].macro_f1.mean()
    local = s[(s.kind == "local") & (s.alpha == 0.1)].groupby("seed").macro_f1.mean().mean()
    add("Overview, N4", "FedAvg macro-F1, alpha 0.5", headline(ov, "N4"), "results_n4/model_scores.csv",
        "mean macro_f1 where kind=federated, method=fedavg, alpha=0.5 (over 3 seeds)", fed)
    add("N4", "centralised macro-F1", next(x for x in A["n4"]["central"] if x["index"] == "macro_f1")["mean"],
        "results_n4/model_scores.csv", "mean macro_f1 where kind=central", central)
    add("N4", "one agency alone, alpha 0.1", next(x for x in A["n4"]["local"] if x["alpha"] == 0.1)["macro_f1_mean"],
        "results_n4/model_scores.csv", "kind=local, alpha=0.1: mean over agencies per seed, then over seeds", local)

    # ------------------------------------------------------------ N5
    g = pd.read_csv(ROOT / "results_n5/eval_grid.csv", dtype={"rho": str, "level": str, "foreign_rate": str})
    sel = lambda v: g[(g.variant == v) & (g.rho == "0.0") & (g.level == "1.0") & (g.foreign_rate == "0.0")].macro_f1.mean()  # noqa: E731
    f1 = lambda v: next(x for x in A["n5"]["macro_f1"] if x["variant"] == v and x["rho"] == "0.0" and x["level"] == 1.0)["mean"]  # noqa: E731
    add("Overview, N5", "macro-F1, all evidence deleted, model C", headline(ov, "N5"), "results_n5/eval_grid.csv",
        "mean macro_f1 where variant=C, rho=0.0, level=1.0, foreign_rate=0.0 (over 3 seeds)", sel("C"))
    add("N5", "macro-F1, all evidence deleted, model A", f1("A"), "results_n5/eval_grid.csv",
        "same rule with variant=A", sel("A"))
    ex = pd.read_csv(ROOT / "results_n5/eval_explain.csv", dtype={"rho": str})
    ter = lambda v: ex[(ex.variant == v) & (ex.rho == "0.0") & (ex.level == 1.0)].ter.mean()  # noqa: E731
    tshow = lambda v: next(x for x in A["n5"]["ter"] if x["variant"] == v and x["rho"] == "0.0" and x["level"] == 1.0)["ter"]  # noqa: E731
    add("N5", "TER, all evidence deleted, model C", tshow("C"), "results_n5/eval_explain.csv",
        "mean ter where variant=C, rho=0.0, level=1.0", ter("C"))
    case = j("results_n5/case_export_check.json")
    add("N5", "CASE files with provenance that conform",
        sum(1 for c in A["n5"]["case_export"] if c["conforms"]), "results_n5/case_export_check.json",
        "count of entries with conforms = true", sum(1 for c in case if c["conforms"]))

    # ------------------------------------------------------------ the newest case analysis
    runs = sorted((ROOT / "app_data/runs").glob("*/result.json"), key=lambda p: p.stat().st_mtime) if (ROOT / "app_data/runs").exists() else []
    if runs:
        rd = runs[-1].parent
        res = json.loads(runs[-1].read_text(encoding="utf-8"))
        tag = f"Case {res['version_id']}"
        dele = pd.read_csv(rd / "deleted.csv")
        rec = pd.read_csv(rd / "recovered.csv")
        graph = json.loads((rd / "case.jsonld").read_text(encoding="utf-8"))["@graph"]
        n_rec_obj = sum(any(f.get("@type") == "observable:RecoveredObjectFacet" for f in n.get("core:hasFacet", [])) for n in graph)
        add(tag + ", step 2", "deleted by the owner", res["scenario"]["n_deleted"], f"{rd.name}/deleted.csv", "rows where owner=suspect",
            int((dele.owner == "suspect").sum()))
        add(tag + ", step 2", "recovered by N3", res["scenario"]["n_recovered"], f"{rd.name}/recovered.csv", "number of rows", len(rec))
        add(tag + ", step 5", "objects in the CASE file", res["case_export"]["objects"], f"{rd.name}/case.jsonld", "length of @graph", len(graph))
        add(tag + ", step 5", "recovered objects", res["scenario"]["n_recovered"], f"{rd.name}/case.jsonld",
            "objects carrying observable:RecoveredObjectFacet", n_rec_obj)
        add(tag + ", step 5", "SHA-256 of the CASE file", res["case_export"]["sha256"], f"{rd.name}/case.jsonld",
            "sha256 of the file bytes", hashlib.sha256((rd / "case.jsonld").read_bytes()).hexdigest())
        p = res["triage"]["proba"]
        add(tag + ", step 3", "predicted class", res["triage"]["pred"], f"{rd.name}/result.json", "class with the largest probability",
            res["triage"]["classes"][max(range(len(p)), key=p.__getitem__)])
    else:
        print("(no case analysed yet: analyse one in the app to trace a case as well)")

    # ------------------------------------------------------------ print and save
    w = max(len(r["number"]) for r in ROWS)
    print(f"{'match':5}  {'screen':26}  {'number':{w}}  {'app shows':>14}  {'from the file':>14}  file")
    for r in ROWS:
        fmt = lambda v: (f"{v:.6f}" if isinstance(v, float) else str(v))[:14]  # noqa: E731
        print(f"{r['match']:5}  {r['screen'][:26]:26}  {r['number']:{w}}  {fmt(r['app_shows']):>14}  {fmt(r['recomputed_from_file']):>14}  {r['file']}")
    out = ROOT / "app_data" / "trace_results.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        wr.writeheader()
        wr.writerows(ROWS)
    bad = [r for r in ROWS if r["match"] != "yes"]
    print(f"\n{len(ROWS) - len(bad)} of {len(ROWS)} numbers match their source file. Full table with the rule for each: {out.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
