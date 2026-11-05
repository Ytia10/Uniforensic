# UniForensic N3 - summary of YOUR run

SQLite engine used to build the databases: 3.53.4 (APSW 3.53.4.0).
Synthetic test: 1920 databases from 480 N1 test cases; 33023 deleted rows, 9630 still present in the bytes.

## Synthetic test databases

| method | reported | precision | recall (all deleted) | recall (present) | ECE |
|---|---|---|---|---|---|
| strict carving | 6456 | 0.989 | 0.193 | 0.663 | - |
| fixed heuristic score | 20721 | 0.463 | 0.290 | 0.995 | 0.315 |
| ranker_raw | 9587 | 0.999 | 0.290 | 0.995 | 0.000 |
| calibrated ranker (N3) | 9581 | 0.999 | 0.290 | 0.994 | 0.000 |
| ranker_cal_t95 | 9599 | 0.998 | 0.290 | 0.995 | 0.000 |

Calibrated ranker, 95% intervals (resampling databases): precision 0.999-1.000, recall (present) 0.993-0.996.
Live rows reported as deleted: 0 for every N3 method (excluded by construction).
Operating point fixed on the calibration split before testing: p >= 0.5 and p >= 0.167 (95% precision).

## Hold-out tests (step 7): worst case over held-out settings

- unseen SQLite condition: precision >= 0.998, recall (present) >= 0.923
- unseen app schema: precision >= 0.949, recall (present) >= 0.627 (worst: web)

## Third-party corpora (never used for training or calibration)

Nemetz record categories: 159 deleted rows, 124 present.

| method | reported | precision | recall (all deleted) | recall (present) | ECE |
|---|---|---|---|---|---|
| strict carving | 31 | 0.968 | 0.189 | 0.234 | - |
| fixed heuristic score | 313 | 0.399 | 0.786 | 1.000 | 0.343 |
| calibrated ranker (N3) | 121 | 0.967 | 0.736 | 0.935 | 0.017 |
| ranker + content feature | 66 | 0.985 | 0.409 | 0.516 | 0.129 |
| ranker without payment-app tables | 129 | 0.837 | 0.679 | 0.863 | 0.045 |
| SQLite Dissect 1.0.0 | 132 | 0.561 | 0.465 | - | - |

NIST SFT-03: 400 deleted rows (4 variants), all present.

| method | reported | precision | recall (all deleted) | recall (present) | ECE |
|---|---|---|---|---|---|
| strict carving | 400 | 1.000 | 1.000 | 1.000 | - |
| fixed heuristic score | 482 | 0.830 | 1.000 | 1.000 | 0.306 |
| calibrated ranker (N3) | 400 | 1.000 | 1.000 | 1.000 | 0.029 |
| ranker + content feature | 400 | 1.000 | 1.000 | 1.000 | 0.000 |
| ranker without payment-app tables | 388 | 1.000 | 0.970 | 0.970 | 0.077 |
| SQLite Dissect 1.0.0 | 596 | 0.000 | 0.000 | - | - |

SQLite Dissect crashed on 8 of the Nemetz record databases and 4 of 4 NIST databases (rows written before a crash are counted).

SQLite Dissect on 240 synthetic test databases: precision 0.095, 6466 live rows reported as deleted, crashed on 42.

## Planted evidence (N1) that was deleted

| pattern | deleted | bytes present | recovered (ranker) | recovered (strict) |
|---|---|---|---|---|
| C1_hacktool_web | 767 | 42.8% | 42.8% | 31.3% |
| F1_scam_sms | 822 | 29.4% | 29.2% | 20.0% |
| F2_mule_inflow | 797 | 21.8% | 21.7% | 16.2% |
| F3_sms_blast | 719 | 42.3% | 42.3% | 30.6% |
| H1_abusive_sms | 761 | 34.0% | 33.9% | 22.5% |
| H2_repeat_calls | 1614 | 22.0% | 22.0% | 16.7% |
| H3_night_calls | 790 | 23.3% | 23.3% | 16.8% |

Figures: figures_n3/fig12_calibration.png, fig13_conditions.png, fig14_third_party.png
