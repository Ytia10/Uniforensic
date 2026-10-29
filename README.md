# UniForensic: a federated and explainable framework for digital forensics

Research prototype for AI-assisted triage of seized smartphones. It checks whether the
model's explanations point at the real evidence, lets agencies train together without
pooling cases, and keeps records recovered from deleted space separate from live data,
all the way into a CASE/UCO evidence file and a printable report.

B.Tech major project, Department of Computer Science and Engineering, Yeshwantrao Chavan
College of Engineering, Nagpur (group G7, guide: Prof. Renuka Kadu).

> **Status:** research prototype. All cases are synthetic, with planted evidence so that
> the right answer is known. It is not a forensic tool and must not be used on real cases.

## What is in this repository

| Part | Question it answers | Code | Results |
|---|---|---|---|
| N1 Validated explanations | Is an explanation right, and when may it be reported? | `ufn1/`, `00_`-`11_*.py` | `results/`, `figures/` |
| N2 Private explanation sharing | What do shared explanation summaries leak across agencies? | `ufn2/`, `n2_*.py` | `results_n2/`, `figures_n2/` |
| N3 SQLite carving | Which deleted SQLite records can be read back, and how sure is the carver? | `n3_team/` (used by the app); `ufn3/`, `n3_*.py` (alternative) | `results_n5/n3_*.json`; `results_n3/` |
| N4 Federated triage | Can agencies train one model over a shared CASE/UCO feature schema? | `ufn4/`, `n4_*.py` | `results_n4/`, `figures_n4/` |
| N5 Recovered-evidence provenance | What happens when a decision rests on recovered records? | `ufn5/`, `n5_*.py` | `results_n5/`, `figures_n5/` |
| Workbench | All five on one phone, live, in the browser | `ufapp/` (FastAPI), `web/` (React) | written to `app_data/` |

The trained models, results and benchmark data are included, so the app runs straight
after cloning. Re-running N1-N5 is optional (see the end).

## Requirements

- Windows 10/11, Linux or macOS
- Python 3.12 (python.org; on Windows tick "Add python.exe to PATH")
- about 3 GB free disk for the Python packages (CPU PyTorch)
- optional: Docker Desktop, instead of a local Python
- not needed: Node.js (the built web page is in `web/dist`) and a GPU

## Start the app

### Windows, without Docker (recommended)

    git clone https://github.com/<your-username>/uniforensic.git
    cd uniforensic
    setup_app.bat        # once: creates .venv, installs pinned packages, runs a quick self-test
    run_app.bat          # every time: starts the app and opens http://localhost:8000

Stop the app with Ctrl+C in its window.

### Linux or macOS, without Docker

    git clone https://github.com/<your-username>/uniforensic.git
    cd uniforensic
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # Linux: avoids a 3 GB CUDA download
    pip install -r requirements_app.txt
    python app_selftest.py --quick
    bash run_app.sh      # then open http://localhost:8000

### With Docker

    docker compose up --build      # first build 5-15 min; then open http://localhost:8000
    docker compose down            # after Ctrl+C

The container is Linux. Live runs that are compared bit for bit with the stored results
(federated training, the DP release, the N3 pipeline) can differ in the last digits from
results produced on Windows; the screens then say "differs". Use the non-Docker route
on the platform the results were produced on for exact matches.

## Check it yourself

| Command | What it proves | Expected |
|---|---|---|
| `python app_selftest.py` | every live screen reproduces the stored N1-N5 results; files match their manifests | `ALL CHECKS PASSED` (2-5 min) |
| `python trace_results.py` | every headline number on screen is recomputed from its result file without the app's code | `25 of 25 numbers match` |
| `python research_checks.py` | the 25 research claims hold on the result files | `25 of 25 claims hold` |
| `setup_test.bat`, then `python ui_test.py` | a browser clicks through every screen | `ALL UI CHECKS PASSED` (about 3 min; `--quick` about 1 min) |

All four write their tables or screenshots to `app_data/` and change nothing else.

## Using the app

1. **Overview:** integrity of every result file, one headline number per part.
2. **Case intake:** pick one of 600 test cases, choose how much evidence the owner deleted, and *Analyse this phone*.
3. **Recovery, Triage, Certainty, Exchange, Report:** the analysis step by step, ending in a CASE/UCO file (with the official validator) and a printable report.
4. **N1-N5 pages:** each part's results; N2, N3 and N4 also run live.

Tick *Show ground truth* (bottom left) to see the planted answer key; leave it off for the examiner's view.

## Re-running the experiments (optional)

Order: N1, then N4, then N2, then N5 (each reads the earlier outputs). With the `.venv` active:

    python run_all.py        # N1 (time depends on the machine; see README_N1.md)
    python n4_run_all.py     # N4, about 20-30 min
    python n2_run_all.py     # N2, about 8-15 min
    python n5_run_all.py     # N5, about 30-60 min

Each part has its own README (`README_N1.md` ... `README_N5.md`). Seeds are fixed, so a
re-run on the same machine reproduces the result files byte for byte. The alternative N3
(`n3_run_all.py`) also needs the third-party SQLite corpora; see `README_N3.md`.

## Repository layout

    ufn1/ ... ufn5/      library code of N1-N5
    n3_team/             N3 carver and classifier used by N5 and the app
    ufapp/               FastAPI backend of the Workbench
    web/src, web/dist    React frontend source and the built page
    data*/ models*/      benchmark cases and trained models
    results*/ figures*/  every result table and figure
    config*.yaml         all settings (seeds, thresholds)
    app_data/            created on first start: live runs, uploads (not in Git)
