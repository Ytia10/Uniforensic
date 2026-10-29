# UniForensic N4 - Federated triage across agencies on one shared schema

N4 lives in the SAME folder as N1 and builds on N1's outputs (data/, models/mlp.pt,
config.yaml). It never modifies N1 files; everything it writes goes to
data_n4/, models_n4/, results_n4/ and figures_n4/.
The full step-by-step guide is the "UniForensic N4 - Implementation Guide" document.

## Quick start (after N1 has been run in this folder)

    # activate the same .venv you used for N1, then:
    pip install -r requirements_n4.txt
    python n4_00_check_env.py

| Script | Guide step | What it does |
|---|---|---|
| n4_00_check_env.py | 2 | Flower + case-utils imports; N1 outputs present; unit checks |
| n4_01_case_export.py | 3 | Export 60 cases to CASE/UCO JSON-LD; official validation; exact feature round trip |
| n4_02_partition.py | 4 | Dirichlet (non-IID) split of 2,400 training cases over 8 agencies, 4 alphas x 3 seeds |
| n4_03_train_baselines.py | 5 | Centralised (= N1 model for seed 2026) and 96 local-only models |
| n4_04_train_federated.py | 6 | FedAvg and FedProx, 40 rounds, for every alpha and seed |
| n4_05_poisoning.py | 7 | One malicious agency (label flip, scaled update) x 4 Flower aggregation rules |
| n4_06_score_models.py | 8 | Macro-F1, oracle pass rate, evidence recall, TER, proxy share for every model |
| n4_07_make_report.py | 9 | 5 figures, results_n4/SUMMARY_N4.md, run_manifest_n4.json |

`python n4_run_all.py` runs all of them (about 20-30 minutes on a laptop CPU).
All settings are in `config_n4.yaml`.
