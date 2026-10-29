"""The live case analysis: one phone, end to end, with every N1-N5 component.

  1. Deletion scenario + 5 real SQLite databases + carving   (N5 scenario, teammate's N3)
  2. Recalibrated confidence + provenance tag                (N5 step 5)
  3. Model inputs for L / A / Af / B / C                      (N5 step 6, N1 features)
  4. Triage with and without the recovered records           (N5 models, N1 MLP recipe)
  5. Explanation + N1's Level-2 acceptance check             (N1)
  6. Recovery-dependence flag + two 90% prediction sets      (N5 steps 9-10)
  7. CASE/UCO export with provenance                         (N4 + N5 step 12)
  8. Consistency check: the live numbers vs the stored N5 run

Every step calls the N1-N5 code itself (nothing re-implemented). The intermediate
tables are written to app_data/runs/<id>/ and read back with N5's own readers, so
the inputs have exactly the types N5 used.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ufn1.common import derive_seed
from ufn1.explainers import run_explainer
from ufn1.metrics import feature_agreement
from ufn1.models import mlp_predict_proba
from ufn1.schema import CLASSES, DECISIVE_FEATURES, FEATURES
from ufn5.calib import apply_isotonic, to_percent
from ufn5.case_provenance import export_with_provenance
from ufn5.common import version_id
from ufn5.conformal import lac_sets
from ufn5.devicedb import TABLES
from ufn5.evaluate import integrated_gradients, recovered_share, to_base
from ufn5.recover import run_case
from ufn5.views import NUMERIC, REC_FEATURES, build, read_recovered, scale, without_recovered

from . import store
from .labels import FEATURE_LABEL, TABLE_LABEL, input_label, input_names
from .settings import CFG, run_dir

VARIANTS = ("L", "A", "Af", "B", "C")
REC_COLUMNS = ["case_id", "table", "timestamp", "direction", "counterparty", "text", "p_raw", "fuzzy_live",
               "strategy", "source_type", "label", "match_id", "match_role", "match_pattern", "answered",
               "duration_s", "domain", "service", "account", "ip_country", "success", "version_id", "amount"]


def _describe(t: str, r) -> str:
    """One line an examiner can read for a recovered record."""
    if t == "sms":
        arrow = "to" if r.direction == "out" else "from"
        return f"SMS {arrow} {r.counterparty}: \"{r.text}\""
    if t == "call":
        kind = "outgoing" if r.direction == "out" else "incoming"
        state = f"{int(float(r.duration_s))} s" if str(r.answered) == "True" else "not answered"
        return f"{kind} call {r.counterparty}, {state}"
    if t == "web":
        return f"visit to {r.domain}"
    if t == "login":
        ok = "successful" if str(r.success) == "True" else "FAILED"
        return f"{ok} {r.service} login as {r.account} from {r.ip_country}"
    return f"payment {r.direction} {r.counterparty}, Rs {float(r.amount):,.2f}"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse(job, case_id: str, level: float, foreign: bool, behaviour: str, variant: str,
            rho: float, seed: int, kernelshap: bool = True) -> dict:
    cfg1, cfg5 = store.cfg1(), store.cfg5()
    keys, cases = store.test_split()
    if case_id not in cases:
        raise ValueError(f"unknown test case {case_id}")
    key, ctab = store.key_of()[case_id], cases[case_id]
    rdir = run_dir(job.id)
    vid = version_id(case_id, level, foreign)

    # ------------------------------------------------ 1. scenario, databases, carving
    job.step("Simulate the phone: plan deletions, build 5 SQLite databases, carve with the N3")
    scfg = dict(cfg5["scenario"])
    if behaviour == "immediate":
        scfg["immediate_share"] = 1.0                 # same random draws, forced outcome
    elif behaviour == "clean-up":
        scfg["immediate_share"] = 0.0
    truth, rec = run_case(case_id, ctab, float(level), bool(foreign), store.n3_classifier(), scfg,
                          cfg5["seed"], cfg1["benchmark"]["window_days"], keep_dir=str(_dbdir(rdir)))
    # every column N5's split-level CSVs have, even if this one case recovered no call or payment
    rec = rec.reindex(columns=[c for c in REC_COLUMNS if c != "version_id"])
    deleted = truth[truth.deleted].drop(columns=["deleted"]).assign(version_id=vid)
    deleted.to_csv(rdir / "deleted.csv", index=False)
    rec.assign(version_id=vid).to_csv(rdir / "recovered.csv", index=False)
    seeded_behaviour = truth.loc[truth.owner == "suspect", "behaviour"].iloc[0]
    job.done_step(f"{int((truth.owner == 'suspect').sum())} artifacts, {int(deleted.owner.eq('suspect').sum())} "
                  f"deleted by the user, {len(rec)} records recovered")

    # ------------------------------------------------ 2-3. confidence + model inputs
    job.step("Recalibrate N3's confidence, tag provenance, build the five model inputs")
    deleted = pd.read_csv(rdir / "deleted.csv", keep_default_na=False, dtype=str)
    recovered = read_recovered(rdir / "recovered.csv")
    versions = pd.DataFrame([{"version_id": vid, "case_id": case_id, "level": float(level), "foreign": bool(foreign)}])
    recal = store.recalibration()
    F, visible = build(ctab, versions, deleted, recovered, recal, cfg5["calibration"]["report_threshold"])
    job.done_step(f"variant {variant}: {36 if variant == 'C' else 27} inputs")

    # ------------------------------------------------ 4. triage
    job.step(f"Triage with model {variant} (rho {rho}, seed {seed}), with and without recovered records")
    model = store.model(variant, rho, seed)
    X = scale(variant, F[variant], store.n1_scaler(), store.rec_scaler())
    X0 = scale(variant, without_recovered(variant, F["L"]), store.n1_scaler(), store.rec_scaler())
    P, P0 = mlp_predict_proba(model, X)[0], mlp_predict_proba(model, X0)[0]
    pred, pred0 = int(P.argmax()), int(P0.argmax())
    job.done_step(f"{CLASSES[pred]} ({P[pred]:.2f}); without recovered records: {CLASSES[pred0]}")

    # ------------------------------------------------ 5. explanation + N1 acceptance
    job.step("Explain (Integrated Gradients) and run N1's acceptance check (KernelSHAP agreement)")
    base = np.zeros(X.shape[1], dtype=np.float32)
    if variant == "C":
        base[27:] = X0[0, 27:]                         # 'nothing recovered' (as N5 step 8)
    ig = integrated_gradients(model, X, np.array([pred]), base, cfg5["explain"]["n_steps"])[0]
    acc = store.acceptance()
    ks, agreement = None, None
    if kernelshap:
        tr = store.npz("train")[f"X_{variant}"]
        rows = np.random.default_rng(derive_seed(cfg1["seed"], "background")).choice(
            len(tr), CFG["kernelshap_background_rows"], replace=False)
        ks = run_explainer("kernelshap", model, X, np.array([pred]), tr[rows],
                           cfg1["explainers"]["kernelshap"], cfg1["seed"], progress=False)[0]
        agreement = feature_agreement(ks, ig, cfg1["scoring"]["k_fixed"])
    reasons = []
    if not all(acc["level1_validated"].get(e, False) for e in (acc["primary"], acc["secondary"])):
        reasons.append("an explainer did not pass N1 Level 1")
    if agreement is None:
        reasons.append("KernelSHAP was not run")
    elif agreement < acc["min_agreement"]:
        reasons.append(f"explainers agree on only {agreement:.0%} of the top 5 (need {acc['min_agreement']:.0%})")
    if P[pred] < acc["tau"]:
        reasons.append(f"model confidence {P[pred]:.2f} is below tau = {acc['tau']}")
    job.done_step("VALIDATED" if not reasons else "NOT VALIDATED: " + "; ".join(reasons))

    # ------------------------------------------------ 6. flag + prediction sets
    job.step("Recovery-dependence flag and the two 90% prediction sets")
    q_full, q_live = store.conformal_q(variant, rho, seed)
    S, S0 = lac_sets(P[None, :], q_full)[0], lac_sets(P0[None, :], q_live)[0]
    flag = pred != pred0
    share = float(recovered_share(ig[None, :])[0]) if variant == "C" else None
    job.done_step(("FLAG: depends on recovered records" if flag else "no flag") +
                  f"; sets {[CLASSES[c] for c in np.where(S)[0]]} / {[CLASSES[c] for c in np.where(S0)[0]]}")

    # ------------------------------------------------ 7. CASE/UCO export
    job.step("Export the case to CASE/UCO with RecoveredObjectFacet + ConfidenceFacet")
    gone = set(deleted[deleted.owner == "suspect"].artifact_id)
    live = {t: d[~d.artifact_id.isin(gone)] for t, d in ctab.items()}
    rec_tabs = {}
    for t in TABLES:
        r = recovered[recovered.table == t].copy()
        for c in NUMERIC.get(t, []):
            r[c] = pd.to_numeric(r[c])
        p_cal = to_percent(apply_isotonic(recal[t], r.p_raw.to_numpy())) if len(r) else np.array([])
        r["confidence_pct"] = np.round(p_cal * 100).astype(int)
        r["artifact_id"] = [f"{case_id}-rec-{t}-{i}" for i in range(len(r))]
        rec_tabs[t] = r
    doc = export_with_provenance(vid, live, rec_tabs)
    (rdir / "case.jsonld").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    job.done_step(f"{len(doc['@graph'])} CASE objects, sha256 {_sha256(rdir / 'case.jsonld')[:16]}...")

    # ------------------------------------------------ 8. consistency with the stored N5 run
    job.step("Consistency check against the stored N5 run")
    consistency = _consistency(case_id, key, level, foreign, behaviour, variant, rho, seed, F, P, P0)
    job.done_step(consistency["summary"])

    # ------------------------------------------------ assemble the result
    names = input_names(variant)
    order = np.argsort(-np.abs(ig), kind="stable")
    base_attr = to_base(ig[None, :])[0]
    vis = [f for f in DECISIVE_FEATURES if bool(visible.iloc[0][f])]
    records = []
    for t in TABLES:
        r = rec_tabs[t]
        for i, row in enumerate(r.itertuples(index=False)):
            records.append({"table": t, "table_label": TABLE_LABEL[t], "timestamp": row.timestamp,
                            "what": _describe(t, row), "p_raw": round(float(row.p_raw), 4),
                            "confidence": int(row.confidence_pct), "strategy": row.strategy,
                            "source_type": row.source_type, "teammate_fuzzy_filter_drops": bool(row.fuzzy_live),
                            "truth": row.label, "truth_role": row.match_role or None})
    result = {
        "run_id": job.id, "case_id": case_id, "version_id": vid, "label_truth": key["label"],
        "tier_truth": key["tier"],
        "scenario": {"level": float(level), "foreign": bool(foreign), "behaviour_requested": behaviour,
                     "behaviour": seeded_behaviour if behaviour == "seeded" else behaviour,
                     "n_artifacts": int((truth.owner == "suspect").sum()),
                     "n_deleted": int(((truth.owner == "suspect") & truth.deleted).sum()),
                     "n_evidence": int((truth.role == "evidence").sum()),
                     "n_evidence_deleted": int(((truth.role == "evidence") & truth.deleted).sum()),
                     "n_foreign": int((truth.owner == "foreign").sum()),
                     "n_recovered": len(recovered),
                     "recovered_by_truth": recovered.label.value_counts().to_dict(),
                     "databases": [f"{case_id}_{t}.db" for t in TABLES if (_dbdir(rdir) / f"{case_id}_{t}.db").exists()]},
        "model": {"variant": variant, "rho": rho, "seed": seed, "n_inputs": len(names)},
        "triage": {"classes": CLASSES, "proba": P.round(6).tolist(), "proba_without_recovered": P0.round(6).tolist(),
                   "pred": CLASSES[pred], "pred_without_recovered": CLASSES[pred0],
                   "confidence": float(P[pred])},
        "flag": {"depends_on_recovered": bool(flag), "recovered_attribution_share": share},
        "sets": {"alpha": cfg5["conformal"]["alpha"], "q_full": q_full, "q_live": q_live,
                 "with_recovered": [CLASSES[c] for c in np.where(S)[0]],
                 "live_only": [CLASSES[c] for c in np.where(S0)[0]]},
        "explanation": {
            "method": "integrated_gradients",
            "top_inputs": [{"name": names[i], "label": input_label(names[i]), "attribution": float(ig[i]),
                            "channel": "recovered" if names[i].startswith("rec_") else "live",
                            "value": float(F[variant].iloc[0][names[i]])} for i in order[:10]],
            "top_base_features": [{"name": FEATURES[i], "label": FEATURE_LABEL[FEATURES[i]],
                                   "attribution_abs": float(base_attr[i]),
                                   "planted_and_visible": FEATURES[i] in vis}
                                  for i in np.argsort(-base_attr, kind="stable")[:5]],
            "acceptance": {"validated": not reasons, "reasons": reasons, "agreement_top5": agreement,
                           "tau": acc["tau"], "min_agreement": acc["min_agreement"],
                           "kernelshap_top5": [names[i] for i in np.argsort(-np.abs(ks), kind="stable")[:5]] if ks is not None else None}},
        "decisive_features": [{"name": f, "label": FEATURE_LABEL[f], "live": float(F["L"].iloc[0][f]),
                               "merged_A": float(F["A"].iloc[0][f]), "weighted_B": float(F["B"].iloc[0][f]),
                               "recovered_channel": float(F["C"].iloc[0][f"rec_{f}"]),
                               "planted": f in key["primary_features"], "visible": f in vis}
                              for f in DECISIVE_FEATURES],
        "recovered_records": records,
        "timeline": _timeline(ctab, deleted, recovered),
        "case_export": {"file": "case.jsonld", "sha256": _sha256(rdir / "case.jsonld"),
                        "objects": len(doc["@graph"])},
        "consistency": consistency,
        "truth": {"label": key["label"], "tier": key["tier"], "planted_features": key["primary_features"],
                  "visible_features": vis, "evidence_patterns": key["evidence_patterns"]},
    }
    (rdir / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return result


def _dbdir(rdir):
    p = rdir / "databases"
    p.mkdir(exist_ok=True)
    return p


def _timeline(ctab, deleted, recovered) -> list[dict]:
    """Records per day: still live, deleted by the user (truth), recovered by N3."""
    gone = set(deleted[deleted.owner == "suspect"].artifact_id)
    a = pd.concat([pd.DataFrame({"day": d.timestamp.str.slice(0, 10), "deleted": d.artifact_id.isin(gone)})
                   for d in ctab.values()])
    out = pd.DataFrame({"live": a[~a.deleted].groupby("day").size(),
                        "deleted": a[a.deleted].groupby("day").size()})
    if len(recovered):
        rec_day = recovered.timestamp.str.slice(0, 10)
        out["recovered"] = rec_day.groupby(rec_day).size()
    else:
        out["recovered"] = 0
    out = out.fillna(0).astype(int).sort_index()
    return [{"day": d, **{k: int(v) for k, v in r.items()}} for d, r in out.iterrows()]


def _consistency(case_id, key, level, foreign, behaviour, variant, rho, seed, F, P, P0) -> dict:
    """Is this live run the SAME as the stored N5 run of this version (if N5 ran it)?"""
    stored_vid = version_id(case_id, 0.0 if key["label"] == "benign" else float(level), foreign)
    in_grid = (behaviour == "seeded" and (key["label"] == "benign" or not foreign)
               and float(level) in [float(x) for x in store.cfg5()["scenario"]["levels"]])
    tv = store.test_versions()
    if not in_grid or stored_vid not in set(tv.version_id):
        return {"in_n5_grid": False, "stored_version": None,
                "summary": "scenario outside N5's evaluated grid (no stored run to compare with)"}
    feat = {}
    for v in VARIANTS:
        s = store.stored_raw(v).loc[stored_vid]
        feat[v] = float(np.abs(F[v].iloc[0][s.index].to_numpy(float) - s.to_numpy(float)).max())
    sp = store.stored_pred(variant, rho, seed)
    i = [k["version_id"] for k in store.npz("test")["keys"]].index(stored_vid)
    dp = float(np.abs(sp["test_proba"][i] - P).max())
    dp0 = float(np.abs(sp["test_proba0"][i] - P0).max())
    same_class = int(sp["test_proba"][i].argmax()) == int(P.argmax())
    ok = max(feat.values()) < 1e-9 and max(dp, dp0) < 1e-5 and same_class
    import sqlite3
    man = store.r5() / "run_manifest_n5.json"
    stored_sqlite = json.loads(man.read_text(encoding="utf-8")).get("sqlite_library") if man.exists() else None
    hint = ""
    if not ok and stored_sqlite and stored_sqlite != sqlite3.sqlite_version:
        hint = (f" The stored run wrote its databases with SQLite {stored_sqlite}, this app uses "
                f"{sqlite3.sqlite_version}; a different library can lay out deleted space differently.")
    return {"in_n5_grid": True, "stored_version": stored_vid, "max_feature_diff": feat,
            "sqlite_live": sqlite3.sqlite_version, "sqlite_stored_run": stored_sqlite,
            "max_proba_diff": dp, "max_proba_diff_without_recovered": dp0, "same_class": same_class,
            "identical": ok,
            "summary": (f"matches stored N5 version {stored_vid} (largest difference: features "
                        f"{max(feat.values()):.0e}, probabilities {max(dp, dp0):.0e}; same class)") if ok else
                       f"DIFFERS from stored N5 version {stored_vid}: largest feature difference "
                       f"{max(feat.values()):.2e}, probability difference {max(dp, dp0):.2e}." + hint}
