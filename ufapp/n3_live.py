"""The teammate's N3, live: (a) their own pipeline, (b) carving a database you upload.

(a) runs n3_team/run_n3_pipeline.run_pipeline() UNCHANGED in a scratch folder, after
    np.random.seed (as N5 step 3 does), and shows their printed output, their JSON
    report and their plots. The trained classifier is compared with the one N5 uses.
(b) carves an uploaded SQLite file with their SQLiteCarver + classifier through
    N5's adapter (exact live-row exclusion), then applies N5's recalibration.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

from ufn5.calib import apply_isotonic, to_percent
from ufn5.devicedb import SCHEMAS
from ufn5.n3_adapter import N3_DIR, carve

from . import store
from .labels import TABLE_LABEL
from .settings import run_dir


def run_teammate_pipeline(job) -> dict:
    seed = store.cfg5()["n3_numpy_seed"]
    rdir = run_dir(job.id)
    job.step("Run the run_n3_pipeline.run_pipeline() unchanged (numpy seeded)")
    code = (f"import sys; sys.path.insert(0, {str(N3_DIR)!r}); import numpy as np; "
            f"np.random.seed({seed}); import run_n3_pipeline as r; r.run_pipeline()")
    proc = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=rdir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        job.log(line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError("the N3's pipeline stopped with an error - see the log")
    rep = json.loads((rdir / "n3_forensic_results_report.json").read_text(encoding="utf-8"))
    plots = sorted(p.name for p in (rdir / "n3_run_output" / "plots").glob("*.png"))
    shutil.rmtree(rdir / "n3_run_output" / "databases", ignore_errors=True)   # 11 scratch DBs
    job.done_step(f"{len(plots)} plots, report written")

    job.step("Compare N3's classifier with the one N5 uses (models_n5/n3_classifier.json)")
    ours = store.n3_classifier()
    cal = rep["calibration"]
    same = (cal["platt_a"] == ours.platt_a and cal["platt_b"] == ours.platt_b and
            [cal["feature_weights"][n] for n in ours.FEATURE_NAMES] == [float(w) for w in ours.weights])
    job.done_step("IDENTICAL" if same else "DIFFERENT (the code or the seed changed)")
    return {"report": rep, "plots": plots, "identical_to_n5": same, "numpy_seed": seed}


def _ro(path):
    """Read-only, immutable connection (documented SQLite URI form, Windows-safe)."""
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def detect_table(path) -> str:
    """Which of the five supported app schemas this SQLite file holds."""
    con = _ro(path)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t, s in SCHEMAS.items():
            if s["table_name"] in tables:
                cols = [r[1] for r in con.execute(f"PRAGMA table_info({s['table_name']})")]
                if cols == s["columns"]:
                    return t
    finally:
        con.close()
    raise ValueError("no supported table found: expected one of " +
                     ", ".join(f"{s['table_name']}({', '.join(s['columns'])})" for s in SCHEMAS.values()))


def carve_upload(job, path: str, original_name: str) -> dict:
    from pathlib import Path
    p = Path(path)
    job.step("Check the file and find its table")
    if p.read_bytes()[:16] != b"SQLite format 3\x00":
        raise ValueError("not a SQLite 3 database (wrong file header)")
    table = detect_table(p)
    job.done_step(f"{SCHEMAS[table]['table_name']} ({TABLE_LABEL[table]})")

    job.step("Carve freeblocks, unallocated space and freelist pages with N3")
    recs = carve(str(p), table, store.n3_classifier())
    job.done_step(f"{len(recs)} distinct candidate records that are not live rows")

    job.step("Recalibrate the confidence (N5 isotonic map for this table)")
    p_raw = np.array([r["p_raw"] for r in recs])
    p_cal = to_percent(apply_isotonic(store.recalibration()[table], p_raw)) if len(recs) else p_raw
    rows = []
    for r, pc in zip(recs, p_cal):
        rows.append({"values": [v if isinstance(v, (int, float, str)) or v is None else repr(v)
                                for v in r["values"]],
                     "valid_row": r["artifact"] is not None, "artifact": r["artifact"],
                     "p_raw": round(float(r["p_raw"]), 4), "confidence": int(round(float(pc) * 100)),
                     "strategy": r["strategy"], "source_type": r["source_type"],
                     "teammate_fuzzy_filter_drops": r["fuzzy_live"]})
    rows.sort(key=lambda x: (-x["valid_row"], -x["confidence"]))
    con = _ro(p)
    n_live = con.execute(f"SELECT COUNT(*) FROM {SCHEMAS[table]['table_name']}").fetchone()[0]
    con.close()
    job.done_step(f"{sum(r['valid_row'] for r in rows)} valid deleted rows recovered")
    return {"file": original_name, "table": table, "table_name": SCHEMAS[table]["table_name"],
            "columns": SCHEMAS[table]["columns"], "live_rows": n_live, "records": rows}
