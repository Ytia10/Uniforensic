# UniForensic N1 - Planted-evidence validation of forensic AI explanations

Part of "A Federated and Explainable Framework for Digital Forensics" (G7, YCCE).
The full step-by-step guide (what each step does, why, and what you should see)
is the "UniForensic N1 - Implementation Guide" document shared with this code.

## Quick start (Python 3.12)

    python -m venv .venv               # Windows: py -3.12 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    # Linux only: pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    python 00_check_env.py

Then run the steps in order (first time: one at a time, reading the output):

| Script | Guide step | What it does |
|---|---|---|
| 00_check_env.py | 2 | Imports, metric self-test, smoke test of all explainers |
| 01_generate_benchmark.py | 3 | Synthetic cases with planted evidence + answer key; freezes thresholds |
| 02_build_features.py | 4 | 27 case-level features, counterfactual features, strict/lenient keys |
| 03_sanity_check.py | 5 | Explainers + metrics on known-answer data |
| 04_train_models.py | 6 | MLP + XGBoost triage models |
| 05_oracle_check.py | 7 | Does the model really use the planted evidence? |
| 06_run_explainers.py | 8 | KernelSHAP, IG, GradientSHAP, LIME, Feature Ablation, random, TreeSHAP |
| 07_score_explanations.py | 9 | Recall/precision vs answer key, disagreement matrices |
| 08_ffidelity.py | 10 | F-Fidelity (ICLR 2025) + naive deletion baseline |
| 09_acceptance_test.py | 11 | Two-level acceptance test, tau calibrated on validation |
| 10_scaffold_attack.py | 12 | Optional: Slack et al. scaffold attack (RQ4) |
| 11_make_report.py | 13 | Figures, results/SUMMARY.md, results/run_manifest.json |

`python run_all.py` runs everything (add `--skip-attack` to skip step 12).
All settings are in `config.yaml`. Verified: a fresh install reproduces every
result file byte for byte on the same machine (seed 2026).




Running everything again later: once you have run Steps 2–13 one by one, python run_all.py repeats the whole pipeline and stops at the first error (--skip-attack leaves out Step 12). Before re-running from Step 3, delete data/, models/, results/ and figures/, so no old file is reused.
