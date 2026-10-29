# UniForensic N5 - Recovery confidence and provenance in triage, explanations and reports

N5 lives in the SAME folder as N1-N4. It reads their outputs (data/, models/,
data_n4/partitions.json, config_n2.yaml, config_n4.yaml) and the teammate's N3 code,
and never modifies any of them; everything it writes goes to data_n5/, models_n5/,
results_n5/ and figures_n5/.

The teammate's N3 code goes UNCHANGED into the folder `n3_team/` (7 files:
carver.py, classifier.py, database_generator.py, evaluator.py, generate_diagrams.py,
run_n3_pipeline.py, sqlite_spec.py). n5_00 checks their md5 against the verified version.

The full step-by-step guide is the "UniForensic N5 - Implementation Guide" document.

## Quick start (after N1 and N4 have been run in this folder)

    # activate the same .venv you used for N1-N4, then:
    pip install -r requirements_n5.txt      # adds only MAPIE 1.5.0
    python n5_00_check_env.py
    python n5_run_all.py                     # about 30-60 min on a 2-core laptop

| Script | Guide step | What it does |
|---|---|---|
| n5_00_check_env.py | 2 | libraries, inputs, N3 md5; self-tests (weighted features = N1, SQLite round trip, one real carve, LAC = MAPIE) |
| n5_01_n3_classifier.py | 3 | trains the teammate's N3 classifier and proves it is bit-identical to their own run_pipeline() |
| n5_02_build_and_carve.py | 4 | every N1 case -> 5 real SQLite databases -> deletions on a timeline -> N3 carving |
| n5_03_calibration.py | 5 | recovery rates, fuzzy live-filter loss, calibration of N3's p on N1 databases + isotonic fix |
| n5_04_features.py | 6 | variants L / A / Af / B / C and the per-version answer key |
| n5_05_train_variants.py | 7 | 30 triage models (5 variants x 2 residue priors x 3 seeds), N1's MLP recipe |
| n5_06_evaluate.py | 8 | macro-F1, false incrimination, evidence recall, TER per deletion level and residue rate |
| n5_07_flag.py | 9 | recovery-dependence flag (counterfactual + attribution) |
| n5_08_conformal.py | 10 | 90% conformal sets with and without recovered records; MAPIE cross-check |
| n5_09_federated.py | 11 | N4 alignment: FedAvg of L/B/C on N4's 8-agency partition |
| n5_10_case_export.py | 12 | N4 alignment: CASE/UCO export with RecoveredObjectFacet + ConfidenceFacet, validated |
| n5_11_dp_release.py | 13 | N2 alignment: DP release of "recovered reliance" with N2's mechanism |
| n5_12_make_report.py | 14 | figures 15-20, example case reports, SUMMARY_N5.md, run_manifest_n5.json |

All settings are in `config_n5.yaml`.
