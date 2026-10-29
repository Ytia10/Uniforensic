# UniForensic N2 - Privacy cost and faithfulness cost of shared explanations

N2 lives in the SAME folder as N1 and N4. It reads their outputs (data/, models/,
data_n4/, models_n4/) and never modifies them; everything it writes goes to
data_n2/, models_n2/, results_n2/ and figures_n2/.
N2 needs NO new libraries: requirements_n4.txt already covers it.
The full step-by-step guide is the "UniForensic N2 - Implementation Guide" document.

## Quick start (after N1 and N4 have been run in this folder)

    # activate the same .venv you used for N1 and N4, then:
    python n2_00_check_env.py

| Script | Guide step | What it does |
|---|---|---|
| n2_00_check_env.py | 2 | N1/N4 outputs present; DP calibration, attack and summary self-tests |
| n2_01_shadow_pool.py | 3 | The attacker's own 4,800 synthetic cases (never N1's) |
| n2_02_explanations.py | 4 | Integrated Gradients with the deployed federated model; clipping bound |
| n2_03_release_attack.py | 5 | Real deployment: leak (tracing attack) and cost (evidence recall) per scheme and epsilon |
| n2_04_distinguishability.py | 6 | Pyrgelis et al. game: leak vs group size, adaptive and passive attackers |
| n2_05_shadow_models.py | 7 | Shadow models (3 recipes) and the overfit positive control |
| n2_06_case_mia.py | 8 | Per-case explanation attacks (variance, L1, L2; threshold and calibrated) |
| n2_07_make_report.py | 9 | 3 figures, results_n2/SUMMARY_N2.md, run_manifest_n2.json |

`python n2_run_all.py` runs all of them (about 8-15 minutes on a laptop CPU).
All settings are in `config_n2.yaml`.
