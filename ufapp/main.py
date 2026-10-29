"""UniForensic Workbench - FastAPI app.

Run:  python -m uvicorn ufapp.main:app --port 8000      (then open http://localhost:8000)
API under /api; the built React app (web/dist) is served at /.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import APP_VERSION, jobs, results, store
from .settings import CFG, FIGURE_DIRS, RUNS, UPLOADS, WEB_DIST

app = FastAPI(title="UniForensic Workbench", version=APP_VERSION)

_integrity: dict = {}
_integrity_lock = threading.Lock()


def _job_or_404(job_id: str) -> jobs.Job:
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "no such job (jobs live in memory; it may predate a restart)")
    return j


@app.exception_handler(store.MissingOutput)
def _missing(_, exc: store.MissingOutput):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ------------------------------------------------------------------ overview
@app.get("/api/health")
def health():
    return {"ok": True, "version": APP_VERSION}


def _integrity_check(force: bool = False) -> dict:
    with _integrity_lock:
        if force or not _integrity:
            from .integrity import check_all
            _integrity.clear()
            _integrity.update(checked_at=time.strftime("%Y-%m-%d %H:%M:%S"), novelties=check_all())
        return dict(_integrity)


@app.get("/api/overview")
def overview(refresh: bool = False):
    R = {n: getattr(results, n)() for n in ("n1", "n2", "n3", "n4", "n5")}
    return {"version": APP_VERSION, "available": {n: r["ok"] for n, r in R.items()},
            "missing": {n: r.get("missing") for n, r in R.items() if not r["ok"]},
            "integrity": _integrity_check(refresh), "headlines": _headlines(R)}


def _pick(rows, **kw):
    for r in rows:
        if all((r.get(k) == v) if not isinstance(v, float) else (r.get(k) is not None and abs(float(r[k]) - v) < 1e-9)
               for k, v in kw.items()):
            return r
    return None


def _headlines(R: dict) -> list[dict]:
    """A few numbers per novelty, read from its own result files."""
    h = []
    if R["n1"]["ok"]:
        a = R["n1"]["acceptance"]["level2_test"]["all offence cases"]
        h.append({"novelty": "N1", "title": "Validated explanations are right",
                  "value": a["precision_of_validated"], "fmt": "pct",
                  "note": f"of explanations marked VALIDATED on test (tau {R['n1']['acceptance']['tau']}); "
                          f"triage macro-F1 {R['n1']['models']['mlp']['macro_f1']:.3f}"})
    if R["n2"]["ok"]:
        none = _pick(R["n2"]["release"], scheme="none")
        dp1 = _pick(R["n2"]["release"], scheme="secagg_dp", eps=1.0)
        if none and dp1:
            h.append({"novelty": "N2", "title": "Tracing attack on shared summaries",
                      "value": dp1["auc"], "fmt": "num3",
                      "note": f"AUC with secure aggregation + DP at epsilon 1 (no protection: {none['auc']:.3f}; 0.5 = guessing)"})
    if R["n3"]["ok"]:
        c = R["n3"]["on_n1_databases"]["calibration"]
        h.append({"novelty": "N3", "title": "N3's carver confidence, recalibrated",
                  "value": c["ece_recalibrated"], "fmt": "num3",
                  "note": f"ECE on N1 databases (as shipped: {c['ece_raw']:.3f}); evidence recovered "
                          f"{R['n3']['on_n1_databases']['recovery_rate_by_role']['evidence']:.1%}"})
    if R["n4"]["ok"]:
        f = _pick(R["n4"]["federated"], method="fedavg", alpha=0.5)
        c = _pick(R["n4"]["central"], index="macro_f1")
        if f and c:
            h.append({"novelty": "N4", "title": "Federated = centralised accuracy",
                      "value": f["macro_f1_mean"], "fmt": "num3",
                      "note": f"FedAvg macro-F1, 8 agencies, alpha 0.5 (centralised {c['mean']:.3f})"})
    if R["n5"]["ok"]:
        a = _pick(R["n5"]["macro_f1"], variant="A", rho="0.0", level=1.0)
        c = _pick(R["n5"]["macro_f1"], variant="C", rho="0.0", level=1.0)
        if a and c:
            h.append({"novelty": "N5", "title": "Triage when ALL evidence was deleted",
                      "value": c["mean"], "fmt": "num3",
                      "note": f"macro-F1, two-channel model C (merging recovered records: {a['mean']:.3f})"})
    return h


# ------------------------------------------------------------------ results pages
@app.get("/api/results/{n}")
def get_results(n: str):
    if n not in ("n1", "n2", "n3", "n4", "n5"):
        raise HTTPException(404, "unknown novelty")
    return getattr(results, n)()


@app.get("/api/figures/{n}/{name}")
def figure(n: str, name: str):
    d = FIGURE_DIRS.get(n)
    p = (d / name) if d else None
    if p is None or p.suffix != ".png" or p.parent != d or not p.exists():
        raise HTTPException(404, "no such figure")
    return FileResponse(p, media_type="image/png")


# ------------------------------------------------------------------ cases
@app.get("/api/models")
def models():
    return {"models": store.available_models(), "default": CFG["default_model"],
            "levels": store.cfg5()["scenario"]["levels"]}


@app.get("/api/cases")
def cases():
    keys, per = store.test_split()
    tv = store.test_versions()
    stored = tv.groupby("case_id").version_id.apply(list).to_dict()
    out = []
    for k in keys:
        c = per[k["case_id"]]
        out.append({"case_id": k["case_id"], "label": k["label"], "tier": k["tier"],
                    "n_artifacts": int(sum(len(d) for d in c.values())),
                    "by_table": {t: int(len(d)) for t, d in c.items()},
                    "stored_versions": stored.get(k["case_id"], [])})
    return {"cases": out}


@app.get("/api/cases/{case_id}")
def case_detail(case_id: str):
    keys, per = store.test_split()
    if case_id not in per:
        raise HTTPException(404, "unknown case")
    k = store.key_of()[case_id]
    arts = []
    for t, d in per[case_id].items():
        for r in d.itertuples(index=False):
            arts.append({"table": t, "timestamp": r.timestamp, "artifact_id": r.artifact_id,
                         "what": _what(t, r), "role": r.role, "pattern": r.pattern_id})
    arts.sort(key=lambda a: a["timestamp"])
    return {"case_id": case_id, "label": k["label"], "tier": k["tier"],
            "planted_features": k["primary_features"], "evidence_patterns": k["evidence_patterns"],
            "distractor_patterns": k["distractor_patterns"], "artifacts": arts}


def _what(t, r) -> str:
    if t == "sms":
        return f"SMS {'to' if r.direction == 'out' else 'from'} {r.counterparty}: \"{r.text}\""
    if t == "call":
        st = f"{int(r.duration_s)} s" if str(r.answered).lower() == "true" else "not answered"
        return f"{'outgoing' if r.direction == 'out' else 'incoming'} call {r.counterparty}, {st}"
    if t == "web":
        return f"visit to {r.domain}"
    if t == "login":
        ok = "successful" if str(r.success).lower() == "true" else "FAILED"
        return f"{ok} {r.service} login as {r.account} from {r.ip_country}"
    return f"payment {r.direction} {r.counterparty}, Rs {float(r.amount):,.2f}"


class AnalyseIn(BaseModel):
    case_id: str
    level: float = 0.0
    foreign: bool = False
    behaviour: str = Field("seeded", pattern="^(seeded|immediate|clean-up)$")
    variant: str = Field("C", pattern="^(L|A|Af|B|C)$")
    rho: float = 0.25
    seed: int = 2026
    kernelshap: bool = True


@app.post("/api/analyse")
def analyse(body: AnalyseIn):
    from .case_pipeline import analyse as run
    if body.level not in [float(x) for x in store.cfg5()["scenario"]["levels"]]:
        raise HTTPException(422, f"level must be one of {store.cfg5()['scenario']['levels']}")
    store.model(body.variant, float(body.rho), body.seed)          # fail fast if missing
    j = jobs.submit("case", run, **{**body.model_dump(), "rho": float(body.rho)})
    return j.public()


# ------------------------------------------------------------------ jobs and runs
@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_or_404(job_id).public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    j = _job_or_404(job_id)
    if j.status != "done":
        raise HTTPException(409, f"job is {j.status}")
    return j.result


@app.get("/api/runs")
def runs():
    out = []
    for f in sorted(RUNS.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        r = json.loads(f.read_text(encoding="utf-8"))
        out.append({"run_id": r["run_id"], "case_id": r["case_id"], "version_id": r["version_id"],
                    "pred": r["triage"]["pred"], "flag": r["flag"]["depends_on_recovered"],
                    "model": r["model"], "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))})
    return {"runs": out}


@app.get("/api/runs/{run_id}")
def run_result(run_id: str):
    p = _safe(RUNS / run_id, "result.json")
    if not p.exists():
        raise HTTPException(404, "no such run")
    return json.loads(p.read_text(encoding="utf-8"))


def _safe(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise HTTPException(400, "bad path")
    return p


@app.get("/api/runs/{run_id}/files/{rel:path}")
def run_file(run_id: str, rel: str):
    p = _safe(RUNS / run_id, rel)
    if not p.is_file():
        raise HTTPException(404, "no such file")
    media = {".png": "image/png", ".jsonld": "application/ld+json", ".json": "application/json",
             ".db": "application/vnd.sqlite3", ".csv": "text/csv"}.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=media, filename=p.name if p.suffix in (".db", ".jsonld", ".csv") else None)


def _validate(job, run_id: str) -> dict:
    job.step("Validate the CASE/UCO file with the official validator (case_utils, CASE 1.5.0)")
    from case_utils.case_validate import validate
    from ufn4.common import load_config_n4
    r = validate(str(RUNS / run_id / "case.jsonld"), case_version=load_config_n4()["case_export"]["case_version"])
    job.done_step(f"conforms = {r.conforms}")
    return {"conforms": bool(r.conforms), "report": r.text[:4000]}


@app.post("/api/runs/{run_id}/validate")
def validate_case(run_id: str):
    if not (RUNS / run_id / "case.jsonld").exists():
        raise HTTPException(404, "this run has no CASE export")
    return jobs.submit("validate", _validate, run_id=run_id).public()


# ------------------------------------------------------------------ N3 (teammate)
@app.post("/api/n3/pipeline")
def n3_pipeline():
    from .n3_live import run_teammate_pipeline
    return jobs.submit("n3_pipeline", run_teammate_pipeline).public()


@app.post("/api/n3/upload")
async def n3_upload(file: UploadFile = File(...)):
    from .n3_live import carve_upload
    limit = int(CFG["upload_max_mb"]) * 1024 * 1024
    dest = UPLOADS / f"{uuid.uuid4().hex[:12]}.db"
    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {CFG['upload_max_mb']} MB")
            out.write(chunk)
    return jobs.submit("n3_upload", carve_upload, path=str(dest), original_name=file.filename or "upload.db").public()


# ------------------------------------------------------------------ N4 federated
class FedIn(BaseModel):
    alpha: float = 0.5
    seed: int = 2026
    method: str = Field("fedavg", pattern="^(fedavg|fedprox)$")
    attack: str = Field("none", pattern="^(none|label_flip|scaled_update)$")
    rule: str = Field("fedavg", pattern="^(fedavg|fedprox|fedmedian|fedtrimmedavg|krum)$")


@app.post("/api/fed/run")
def fed_run(body: FedIn):
    from .fed_live import run
    return jobs.submit("fed", run, **body.model_dump()).public()


# ------------------------------------------------------------------ N2 privacy
@app.post("/api/privacy/prepare")
def privacy_prepare():
    from .privacy_live import prepare
    return jobs.submit("privacy", prepare).public()


@app.get("/api/privacy/release")
def privacy_release(eps: str = "1", repeat: int = 0):
    from .privacy_live import do_release, ready
    if not ready():
        raise HTTPException(409, "prepare the privacy demo first")
    e = math.inf if eps in ("inf", "none", "null") else float(eps)
    return do_release(e, repeat)


# ------------------------------------------------------------------ the web app
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/")
    def no_web():
        return JSONResponse({"detail": "web/dist not found - the API is running; build the web app "
                                       "(cd web && npm ci && npm run build) or use the shipped build."})
