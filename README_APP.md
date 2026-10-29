# UniForensic Workbench - end-to-end demonstration app

A FastAPI backend (`ufapp/`) and a React + TypeScript frontend (`web/`) over the
N1-N5 code and outputs in THIS folder. The app reads N1-N5 (and the teammate's N3
in `n3_team/`) and never changes them; it writes only to `app_data/`.
The full guide is the "UniForensic Workbench - Implementation Guide" document.

## Before you start

N1-N5 must have been run in this folder (their `data*`, `models*`, `results*`,
`figures*` folders exist). The Overview page and `app_selftest.py` tell you exactly
what is missing, if anything.

## Option A - Docker (Windows 10/11 with Docker Desktop)

    docker compose up --build          # first build: 5-15 min (downloads Python packages)
    # open http://localhost:8000       stop with Ctrl+C, then: docker compose down
    docker compose run --rm workbench python app_selftest.py     # self-test: run it once

Live runs are compared with the outputs N1-N5 wrote on YOUR machine. If you ran N1-N5
natively on Windows, the container is another platform (Linux, maybe another SQLite
version, CPU PyTorch) and a live run can differ from the stored one in the last digits.
If self-test groups 3 or 4 report differences in Docker, use Option B for the demo.

## Option B - without Docker (Windows)

    setup_app.bat                      # once: installs requirements_app.txt into .venv, runs the quick self-test
    run_app.bat                        # starts the app and opens http://localhost:8000

## What runs where

| Screen | Live computation | From stored results |
|---|---|---|
| Overview | integrity check of N1-N5 files (sha256 / md5) | headline numbers |
| 1-6 Case workbench | phone databases, teammate's N3 carving, triage, IG + KernelSHAP, flag, prediction sets, CASE export + validator, report | conformal thresholds of N5 step 10 |
| N1 | - | benchmark, acceptance gate, scaffold attack, figures |
| N2 | DP release of recovered reliance (N2 x N5) | leakage vs epsilon, figures |
| N3 | the teammate's pipeline; carving an uploaded SQLite file | their benchmark, calibration on N1 databases |
| N4 | FedAvg / FedProx / poisoning run, round by round | accuracy vs skew, convergence, poisoning, figures |
| N5 | - | variants, false incrimination, flag, coverage, figures |

Every live run that matches a stored experiment is compared with it: case analyses
with N5's stored versions, federated runs with N4's stored models (bit for bit), DP
releases with `results_n5/dp_release.csv`, the teammate's pipeline with N5's classifier.

## Testing (full steps: the "UniForensic Workbench - Testing Guide" document)

    python app_selftest.py             # backend: every live run reproduces the stored N1-N5 results
    python trace_results.py            # every headline number traced back to its result file
    setup_test.bat                     # once: installs Playwright + Chromium for the browser test
    python ui_test.py                  # frontend: clicks through every screen (--quick: 1 min)

With Docker, run the first two inside the container (`docker compose run --rm workbench python app_selftest.py`)
and the browser test from Windows against the running app: `python ui_test.py --url http://localhost:8000`.

## Rebuilding the web page (only if you change `web/src`)

    cd web
    npm ci
    npm run build                      # writes web/dist, which the backend serves

The zip ships `web/dist` prebuilt, so Node.js is NOT needed to run the app.
