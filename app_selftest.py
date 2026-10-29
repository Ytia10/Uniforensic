"""Self-test of the UniForensic Workbench: every live demo must reproduce the stored run.

Runs the app in-process (FastAPI TestClient, no browser, no server) and checks:
  1. all N1-N5 outputs are present and unchanged (integrity against their manifests)
  2. every results page loads
  3. live case analyses == the stored N5 versions (features, probabilities, class)
  4. a live FedAvg run == N4's stored model, bit for bit (and one poisoning run)
  5. the live DP release == results_n5/dp_release.csv for every epsilon and noise draw
  6. the teammate's pipeline, run live, trains the classifier N5 uses
  7. carving an uploaded database == the records the case analysis recovered
  8. the live CASE/UCO export passes the official CASE 1.5.0 validator
Run:  python app_selftest.py            (about 3-5 minutes)    --quick skips 4, 6 and 8
"""
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message=".*httpx.*")     # starlette's notice about its test client
from fastapi.testclient import TestClient                   # noqa: E402

from ufn1.common import ROOT
from ufapp.main import app

QUICK = "--quick" in sys.argv
c = TestClient(app)
failures = []


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]  {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def wait(job, timeout=900):
    t0 = time.time()
    while True:
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            if j["status"] == "error":
                print("\n".join(j["lines"][-15:]))
            return j
        if time.time() - t0 > timeout:
            raise TimeoutError(job["id"])
        time.sleep(0.3)


print("1. Inputs and integrity")
ov = c.get("/api/overview?refresh=true").json()
check("N1-N5 outputs present", all(ov["available"].values()), str(ov.get("missing")))
for n, r in ov["integrity"]["novelties"].items():
    check(f"{n} files match their manifest", r["status"] == "ok", f"{r['ok']}/{r['checked']} ok")

print("2. Results pages")
for n in ("n1", "n2", "n3", "n4", "n5"):
    r = c.get(f"/api/results/{n}").json()
    check(f"/api/results/{n}", r["ok"])

print("3. Live case analysis == stored N5 run")
versions = [("TE00150", 1.0, False), ("TE00000", 0.0, True), ("TE00320", 0.5, False),
            ("TE00472", 0.25, False), ("TE00068", 0.0, False), ("TE00599", 0.1, False)]
runs = []
for cid, lv, fo in versions:
    j = wait(c.post("/api/analyse", json={"case_id": cid, "level": lv, "foreign": fo}).json())
    r = c.get(f"/api/jobs/{j['id']}/result").json()
    runs.append(r)
    check(f"{cid} level {lv} residue {fo}", r["consistency"].get("identical", False), r["consistency"]["summary"])

if not QUICK:
    print("4. Live federated training == N4's stored models")
    for body in ({"alpha": 0.5, "seed": 2026},
                 {"alpha": 0.5, "seed": 2027, "attack": "label_flip", "rule": "fedmedian"}):
        j = wait(c.post("/api/fed/run", json=body).json())
        r = c.get(f"/api/jobs/{j['id']}/result").json()
        check(f"FedAvg {body}", r["identical_to_stored"] is True, r["stored_model"])

print("5. Live DP release == results_n5/dp_release.csv")
wait(c.post("/api/privacy/prepare").json())
d = pd.read_csv(ROOT / "results_n5" / "dp_release.csv")
worst = 0.0
for (eps, rep), g in d.groupby(["eps", "repeat"]):
    out = c.get(f"/api/privacy/release?eps={'inf' if eps == float('inf') else eps}&repeat={int(rep)}").json()
    for row in out["rows"]:
        s = g[g["class"] == row["class"]].iloc[0]
        worst = max(worst, abs(s.released_reliance - row["released_reliance"]))
check(f"{len(d)} released values", worst < 1e-12, f"max difference {worst:.1e}")

if not QUICK:
    print("6. Teammate's N3 pipeline, live")
    j = wait(c.post("/api/n3/pipeline").json(), timeout=600)
    r = c.get(f"/api/jobs/{j['id']}/result").json()
    check("their classifier == the one N5 uses", r["identical_to_n5"], f"{len(r['plots'])} plots")

print("7. Upload a database and carve it")
r0 = runs[0]
db = r0["scenario"]["databases"][0]
blob = c.get(f"/api/runs/{r0['run_id']}/files/databases/{db}").content
j = wait(c.post("/api/n3/upload", files={"file": (db, blob, "application/octet-stream")}).json())
u = c.get(f"/api/jobs/{j['id']}/result").json()
want = sum(1 for x in r0["recovered_records"] if x["table"] == u["table"])
got = sum(1 for x in u["records"] if x["valid_row"])
check(f"{db}: valid recovered rows", got == want, f"{got} uploaded vs {want} in the case analysis")

if not QUICK:
    print("8. CASE/UCO export of a live run")
    j = wait(c.post(f"/api/runs/{r0['run_id']}/validate").json(), timeout=600)
    r = c.get(f"/api/jobs/{j['id']}/result").json()
    check("official CASE 1.5.0 validator", r["conforms"])

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)
