"""Automated browser test of the Workbench: clicks through every screen like a person would.

It starts its own copy of the app on port 8765 (unless you pass --url), opens a real
Chromium browser without a window, and checks that each screen loads without errors and
shows the same numbers the backend computed. Screenshots of every screen are saved to
app_data/ui_test/ so you can look at what was tested.

Setup (once):   pip install -r requirements_test.txt
                python -m playwright install chromium
Run:            python ui_test.py              (about 4-6 minutes)
                python ui_test.py --quick      (skips the slow live runs: about 1 minute)
                python ui_test.py --show       (opens a visible browser window, to watch)
                python ui_test.py --url http://localhost:8000   (test an app you already started)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUICK = "--quick" in sys.argv
SHOW = "--show" in sys.argv
URL = sys.argv[sys.argv.index("--url") + 1].rstrip("/") if "--url" in sys.argv else "http://127.0.0.1:8765"
SHOTS = ROOT / "app_data" / "ui_test"
FAILS: list[str] = []
CASE = "TE00150"   # fraud case used in the demo script: 100% of the evidence deleted


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]  {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def api(path: str):
    with urllib.request.urlopen(URL + path, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_for_server(proc=None, seconds: int = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            api("/api/health")
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    from playwright.sync_api import sync_playwright

    proc = None
    if "--url" not in sys.argv:
        print(f"Starting the app on {URL} ...")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "ufapp.main:app", "--port", URL.rsplit(":", 1)[1]],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if not wait_for_server(proc):
        print("The app did not start. Run it by hand (run_app.bat) to see the error.")
        return 1
    SHOTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    try:
        with sync_playwright() as p:
            br = p.chromium.launch(headless=not SHOW)
            pg = br.new_page(viewport={"width": 1440, "height": 1000})
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.set_default_timeout(120_000)

            def goto(page: str, wait_text: str | None = None):
                pg.goto(f"{URL}/#/{page}")
                if wait_text:
                    pg.get_by_text(wait_text, exact=False).first.wait_for()
                pg.wait_for_timeout(1200)

            def shot(name: str):
                pg.screenshot(path=str(SHOTS / f"{name}.png"), full_page=False)

            # ---------------------------------------------------------------- 1
            print("1. Every screen opens without errors (before any case is analysed)")
            pg.goto(URL)
            pg.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
            for page, text in [("overview", "Integrity of the results"), ("intake", "Test cases"), ("recovery", "No phone is on the bench yet"),
                               ("n1", "Level 1"), ("n2", "Privacy against faithfulness"), ("n3", "N3's benchmark"),
                               ("n4", "Accuracy by case-mix skew"), ("n5", "Triage accuracy as evidence is deleted")]:
                n0 = len(errors)
                goto(page, text)
                check(f"#/{page} shows '{text}'", True)
                check(f"#/{page} has no browser errors", len(errors) == n0, "; ".join(errors[n0:])[:200])

            # ---------------------------------------------------------------- 2
            print("2. Overview shows the backend's numbers")
            ov = api("/api/overview")
            goto("overview", "Five contributions")
            body = pg.inner_text("main")
            items = ov["integrity"]["novelties"].values()
            ok_n, all_n = sum(i["ok"] for i in items), sum(i["checked"] for i in items)
            check("integrity count on screen", f"{ok_n} / {all_n}" in body, f"{ok_n} / {all_n}")
            check("every novelty's files unchanged", all(i["status"] == "ok" for i in items))
            for h in ov["headlines"]:
                v = f"{h['value'] * 100:.1f}%" if h["fmt"] == "pct" else f"{h['value']:.3f}"
                check(f"{h['novelty']} headline {v} on screen", v in body)
            shot("01_overview")

            # ---------------------------------------------------------------- 3
            print(f"3. Analyse case {CASE} through the screens (100% of the evidence deleted)")
            goto("intake", "Test cases")
            pg.get_by_placeholder("Search case id").fill(CASE)
            pg.get_by_role("cell", name=CASE, exact=True).click()
            pg.get_by_role("heading", name="Deletion scenario and model").wait_for()
            pg.locator("label:has-text('Share of planted evidence deleted') select").select_option("1")
            pg.get_by_role("button", name="Analyse this phone").click()
            pg.get_by_role("button", name="Next: recovery").wait_for(timeout=180_000)
            shot("02_intake")
            run = api("/api/runs")["runs"][0]
            res = api(f"/api/runs/{run['run_id']}")
            check("analysis finished and was stored", res["case_id"] == CASE, res["version_id"])
            s = res["scenario"]

            pg.get_by_role("button", name="Next: recovery").click(); pg.wait_for_timeout(1500)
            body = pg.inner_text("main")
            check("step 2 shows records, deleted and recovered counts",
                  all(str(x) in body for x in (s["n_artifacts"], s["n_deleted"], s["n_recovered"])),
                  f"{s['n_artifacts']}, {s['n_deleted']}, {s['n_recovered']}")
            check("step 2 lists one table row per recovered record",
                  pg.locator("main tbody tr:visible").count() == len(res["recovered_records"]), f"{len(res['recovered_records'])} rows")
            shot("03_recovery")

            pg.get_by_role("button", name="Next: triage").click(); pg.wait_for_timeout(2500)
            body = pg.inner_text("main")
            conf = f"{res['triage']['confidence'] * 100:.1f}%"
            check("step 3 shows the predicted class and probability", res["triage"]["pred"] in body and conf in body, f"{res['triage']['pred']} {conf}")
            verdict = "VALIDATED" if res["explanation"]["acceptance"]["validated"] else "NOT VALIDATED"
            check("step 3 shows N1's verdict", verdict in body, verdict)
            check("step 3 draws the explanation chart", pg.locator(".recharts-bar-rectangle").count() > 0)
            shot("04_triage")

            pg.get_by_role("button", name="Next: certainty").click(); pg.wait_for_timeout(1500)
            body = pg.inner_text("main")
            flag_text = "This conclusion depends on recovered records." if res["flag"]["depends_on_recovered"] else "This conclusion stands on live records alone."
            check("step 4 shows the recovery flag", flag_text in body, flag_text)
            for c in res["sets"]["with_recovered"] + res["sets"]["live_only"]:
                check(f"step 4 prediction sets include {c}", c in body)
            check("step 4: live result matches the stored N5 run", "matches the stored run" in body, res["consistency"]["summary"][:90])
            shot("05_certainty")

            # ---------------------------------------------------------------- 4
            print("4. Ground truth switch hides and shows the answer key")
            check("truth hidden while the switch is off", "truth:" not in body)
            pg.locator("aside input[type=checkbox]").check(); pg.wait_for_timeout(800)
            goto("triage")
            check("truth shown after switching on", f"truth: {res['truth']['label']}" in pg.inner_text("main"), res["truth"]["label"])
            pg.locator("aside input[type=checkbox]").uncheck()

            # ---------------------------------------------------------------- 5
            print("5. Evidence exchange and report")
            goto("exchange", "Validate and download")
            body = pg.inner_text("main")
            check("step 5 shows the CASE object count and SHA-256",
                  str(res["case_export"]["objects"]) in body and res["case_export"]["sha256"][:16] in body)
            if not QUICK:
                pg.get_by_role("button", name="Run the CASE validator (about 1-1.5 min)").click()
                pg.get_by_text("conforms = True").wait_for(timeout=400_000)
                check("official CASE validator says the file conforms", "conforms" in pg.inner_text("main"))
            shot("06_exchange")
            goto("report", "Finding")
            body = pg.inner_text("main")
            check("step 6 report names the case, class and file hash",
                  CASE in body and res["triage"]["pred"] in body and res["case_export"]["sha256"] in body)
            shot("07_report")

            # ---------------------------------------------------------------- 6
            if QUICK:
                print("6-8. Live novelty runs skipped (--quick)")
            else:
                print("6. N2: live privacy release")
                goto("n2", "Privacy against faithfulness")
                pg.get_by_role("button", name="Prepare (about 20-40 s)").click()
                pg.get_by_text("Mean absolute error").wait_for(timeout=300_000)
                check("N2 live release appears", True)
                pg.locator("input[type=range]").focus()
                for _ in range(5):
                    pg.keyboard.press("ArrowRight")
                pg.wait_for_timeout(1000)
                check("N2 noise draw slider moves", "Noise draw 6 of 20" in pg.inner_text("main"))
                shot("08_n2")

                print("7. N3: teammate's pipeline and a database upload")
                goto("n3", "N3's benchmark")
                pg.get_by_role("button", name="Run their pipeline").click()
                pg.get_by_text("Classifier trained by their code").wait_for(timeout=300_000)
                check("their classifier is identical to the one N5 uses", "identical" in pg.inner_text("main"))
                db = ROOT / "app_data" / "runs" / run["run_id"] / "databases" / f"{CASE}_sms.db"
                pg.locator("input[type=file]").set_input_files(str(db))
                pg.locator("p", has_text="deleted rows recovered").wait_for(timeout=120_000)
                n_sms = sum(1 for r in res["recovered_records"] if r["table"] == "sms")
                check("uploaded SMS database: same recovered count as the analysis",
                      f"{n_sms} deleted rows recovered" in pg.inner_text("main"), f"{n_sms}")
                shot("09_n3")

                print("8. N4: train the federation live")
                goto("n4", "Accuracy by case-mix skew")
                pg.get_by_role("button", name="Train (about 10-40 s)").click()
                pg.get_by_text("identical to fed_fedavg_a0.5_s2026.pt").first.wait_for(timeout=300_000)
                pg.get_by_text(re.compile(r"fraud F1 \d")).wait_for(timeout=300_000)
                check("live FedAvg model identical to N4's stored model", True)
                shot("10_n4")

            # ---------------------------------------------------------------- 9
            print("9. Layout")
            pg.set_viewport_size({"width": 390, "height": 900})
            for page in ("overview", "triage"):
                goto(page); pg.wait_for_timeout(800)
                over = pg.evaluate("() => { const m = document.querySelector('#main'); return m.scrollWidth - m.clientWidth; }")
                check(f"#/{page} fits a 390 px phone screen", over <= 1, f"{over} px too wide" if over > 1 else "")
            pg.set_viewport_size({"width": 1440, "height": 1000})
            goto("n5", "Triage accuracy")
            check("N5 page draws its charts", pg.locator(".recharts-line-curve").count() >= 10)
            shot("11_n5")
            check("no browser errors during the whole test", not errors, "; ".join(errors)[:300])
            br.close()
    except Exception as e:  # a step could not be done at all: report it as a failure, not a crash
        msg = str(e).splitlines()[0][:300]
        print(f"  [FAIL]  the test stopped: {msg}")
        FAILS.append("test stopped early: " + msg)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    m, sec = divmod(int(time.time() - t_start), 60)
    print(f"\nScreenshots: {SHOTS.relative_to(ROOT)}")
    print("ALL UI CHECKS PASSED" if not FAILS else f"{len(FAILS)} UI CHECK(S) FAILED: " + "; ".join(FAILS), f"({m} min {sec} s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
