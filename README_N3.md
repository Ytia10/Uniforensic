# UniForensic N3 - Calibrated recovery of deleted SQLite records

N3 lives in the SAME folder as N1, N2 and N4. It reads N1's benchmark (data/,
config.yaml) and never modifies anything of N1, N2 or N4; everything it writes goes
to data_n3/, models_n3/, results_n3/ and figures_n3/.
New libraries: apsw (bundles its own SQLite, so every computer builds identical
databases) and sqlite-dissect (the external baseline) - see requirements_n3.txt.
The full step-by-step guide is the "UniForensic N3 - Implementation Guide" document.

## Quick start (after N1 has been run in this folder)

    # activate the same .venv you used for N1/N2/N4, then:
    pip install -r requirements_n3.txt
    python n3_00_check_env.py
    git clone https://github.com/SecurityRonin/sqlite-forensic.git ext_sqlite_forensic
    python n3_run_all.py --source ext_sqlite_forensic

| Script | Guide step | What it does |
|---|---|---|
| n3_00_check_env.py | 2 | libraries; reader == SQLite on 12 databases; carver and journal self-tests |
| n3_01_build_databases.py | 3 | N1 cases -> SMS, call-log, Chrome-history and payment databases; deletions; truth log |
| n3_02_carve.py | 4 | carve every database; label each recovered record with the truth |
| n3_03_train_ranker.py | 5 | train the ranker; calibrate on a separate split; fix the operating points |
| n3_04_evaluate.py | 6 | test-set numbers: 4 methods, per condition, per app, deleted planted evidence |
| n3_05_shift.py | 7 | leave-one-condition-out and leave-one-app-out |
| n3_06_get_corpora.py | 8 | copy + MD5-verify the Nemetz corpus and NIST CFReDS SFT-03 |
| n3_07_corpora.py | 9 | third-party test incl. ablations and SQLite Dissect |
| n3_08_export_for_n5.py | 10 | recovered records + confidence for N5 |
| n3_09_make_report.py | 11 | 3 figures, results_n3/SUMMARY_N3.md, run_manifest_n3.json |

`python n3_run_all.py --source <folder>` runs all of them (about 20-40 minutes on a
laptop CPU; SQLite Dissect alone takes about half of that). All settings are in
`config_n3.yaml`.
