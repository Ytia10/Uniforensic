"""Integrity: do the files on disk still match what each novelty recorded when it ran?

N1  models/mlp.pt + xgb.json vs results/model_performance.json 'model_hashes';
    the acceptance thresholds vs results/frozen_thresholds.json (frozen before any result)
N2  configs + shadow models vs results_n2/run_manifest_n2.json
N3  the teammate's 7 files vs the md5 N5 was verified with
N4  configs + every model vs results_n4/run_manifest_n4.json
N5  every listed result/model/data file vs results_n5/run_manifest_n5.json
A changed file is reported, not "fixed": the examiner decides what to do.
"""
from __future__ import annotations

import hashlib

from ufn1.common import ROOT, load_json, sha256_file, sha256_of

from .store import cfg1


def _check(items: dict, base) -> dict:
    ok, changed, missing = 0, [], []
    for rel, want in items.items():
        # manifests written on Windows store paths with "\" (e.g. "results_n5\\eval_grid.csv");
        # on Linux (Docker) that is one file name, so normalise to "/", which both systems accept
        p = base / rel.replace("\\", "/")
        if not p.exists():
            missing.append(rel)
        elif sha256_file(p) != want:
            changed.append(rel)
        else:
            ok += 1
    return {"checked": len(items), "ok": ok, "changed": changed, "missing": missing,
            "status": "ok" if ok == len(items) else ("missing" if missing and not changed else "changed")}


def _manifest(path):
    return load_json(path) if path.exists() else None


def check_all() -> dict:
    out = {}
    perf = _manifest(ROOT / "results" / "model_performance.json")
    frozen = _manifest(ROOT / "results" / "frozen_thresholds.json")
    if perf is None:
        out["N1"] = {"status": "missing", "checked": 0, "ok": 0, "changed": [], "missing": ["results/model_performance.json"]}
    else:
        r = _check(perf["model_hashes"], ROOT / cfg1()["paths"]["models"])
        thr_ok = frozen is not None and frozen["sha256"] == sha256_of(cfg1()["acceptance"])
        r["checked"] += 1
        r["ok"] += int(thr_ok)
        if not thr_ok:
            r["changed"].append("config.yaml acceptance thresholds")
            r["status"] = "changed"
        out["N1"] = r

    for n, mdir in (("N2", "models_n2"), ("N4", "models_n4")):
        m = _manifest(ROOT / f"results_{n.lower()}" / f"run_manifest_{n.lower()}.json")
        if m is None:
            out[n] = {"status": "missing", "checked": 0, "ok": 0, "changed": [],
                      "missing": [f"results_{n.lower()}/run_manifest_{n.lower()}.json"]}
            continue
        items = {f"{mdir}/{k}": v for k, v in m["models"].items()}
        for key, fname in (("config_n1_sha256", "config.yaml"), ("config_n2_sha256", "config_n2.yaml"),
                           ("config_n4_sha256", "config_n4.yaml")):
            if key in m:
                items[fname] = m[key]
        out[n] = _check(items, ROOT)

    from ufn5.n3_adapter import N3_DIR, N3_FILES
    ok = [f for f, h in N3_FILES.items() if (N3_DIR / f).exists()
          and hashlib.md5((N3_DIR / f).read_bytes()).hexdigest() == h]
    miss = [f for f in N3_FILES if not (N3_DIR / f).exists()]
    out["N3"] = {"checked": len(N3_FILES), "ok": len(ok), "missing": miss,
                 "changed": [f for f in N3_FILES if f not in ok and f not in miss],
                 "status": "ok" if len(ok) == len(N3_FILES) else "changed", "note": "teammate's code, md5"}

    m5 = _manifest(ROOT / "results_n5" / "run_manifest_n5.json")
    out["N5"] = (_check(m5["files"], ROOT) if m5 else
                 {"status": "missing", "checked": 0, "ok": 0, "changed": [], "missing": ["results_n5/run_manifest_n5.json"]})
    return out