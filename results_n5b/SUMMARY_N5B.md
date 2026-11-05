# UniForensic N5b - summary of this run (N5 with the alternative N3 carver)

## Alternative N3 (ufn3) on N1's databases

- Carver: ufn3 (alternative N3); ranker models_n3\ranker.joblib (sha256 c34df78a3481a6d0...), isotonic calibration, report threshold p >= 0.5.
- Same carver on public corpora (n3_07, not used for training): nemetz (records (07+0C+0D+0E)): precision 0.967, recall of present records 0.935, ECE 0.017; nist_sft03 (all 4 variants): precision 1.000, recall of present records 1.000, ECE 0.029.

### Both carvers on the same public corpora (n5b_01b; p >= 0.5)

| corpus | method | deleted | present | reported | correct | precision | recall (all) | recall (present) | ECE |
|---|---|---|---|---|---|---|---|---|---|
| nemetz | teammate_n3 | 159 | 124 | 9 | 7 | 0.778 | 0.044 | 0.056 | 0.080 |
| nemetz | ranker_cal | 159 | 124 | 121 | 117 | 0.967 | 0.736 | 0.935 | 0.017 |
| nemetz | sqlite_dissect | 159 | 124 | 132 | 74 | 0.561 | 0.465 | nan | nan |
| nist_sft03 | teammate_n3 | 400 | 400 | 0 | 0 | nan | 0.000 | 0.000 | 0.035 |
| nist_sft03 | ranker_cal | 400 | 400 | 400 | 400 | 1.000 | 1.000 | 1.000 | 0.029 |
| nist_sft03 | sqlite_dissect | 400 | 400 | 596 | 0 | 0.000 | 0.000 | nan | nan |

- Same planted + deleted records as N5: True (143544 test rows).
- Deleted planted evidence recovered: 41.7% (background 69.0%, foreign residue 47.9%).
- Recovered records that are fabrications ('wrong'): 1.48%.
- ECE of N3's p: on N1 test databases 0.0029; after isotonic recalibration 0.0006.
- Correctly recovered deleted records the fuzzy live filter would discard: 63.3% (planted evidence: 19.2%).

## Triage (macro-F1, mean +- SD over 3 seeds, rho = 0, clean benign devices)

| variant | deleted 0% | deleted 10% | deleted 25% | deleted 50% | deleted 100% |
|---|---|---|---|---|---|
| L | 0.923 +- 0.004 | 0.914 +- 0.007 | 0.887 +- 0.002 | 0.801 +- 0.009 | 0.107 +- 0.009 |
| A | 0.919 +- 0.002 | 0.910 +- 0.001 | 0.894 +- 0.003 | 0.850 +- 0.006 | 0.619 +- 0.003 |
| Af | 0.918 +- 0.004 | 0.911 +- 0.003 | 0.898 +- 0.003 | 0.845 +- 0.004 | 0.621 +- 0.003 |
| B | 0.919 +- 0.001 | 0.911 +- 0.002 | 0.894 +- 0.005 | 0.850 +- 0.008 | 0.622 +- 0.004 |
| C | 0.891 +- 0.003 | 0.890 +- 0.004 | 0.881 +- 0.002 | 0.861 +- 0.007 | 0.781 +- 0.000 |
| A, N1 model unchanged | 0.929 | 0.916 | 0.888 | 0.844 | 0.580 |

## Explanations (TER, rho = 0)

| variant | 0% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| L | 0.830 | 0.789 | 0.736 | 0.662 | 0.002 |
| A | 0.853 | 0.824 | 0.788 | 0.731 | 0.455 |
| B | 0.853 | 0.824 | 0.790 | 0.731 | 0.458 |
| C | 0.804 | 0.807 | 0.793 | 0.791 | 0.848 |

## Innocent devices triaged as offences

| variant | rho | clean | with foreign residue |
|---|---|---|---|
| L | 0.0 | 0.202 | 0.202 |
| L | 0.25 | 0.202 | 0.202 |
| A | 0.0 | 0.256 | 0.691 |
| A | 0.25 | 0.253 | 0.667 |
| B | 0.0 | 0.256 | 0.691 |
| B | 0.25 | 0.253 | 0.671 |
| C | 0.0 | 0.233 | 0.784 |
| C | 0.25 | 0.258 | 0.642 |
| A | N1 | 0.113 | 0.560 |

## Recovery-dependence flag

- Residue false incriminations caused by recovered records (C, rho 0): 84.1%; all carry the counterfactual flag (exact test).
- Attribution flag (C) vs counterfactual: agreement 0.862, catches 80.7% of recovery-caused false incriminations.
- Workload (correct offence decisions flagged, C rho 0): 0%: 0.013, 10%: 0.031, 25%: 0.076, 50%: 0.184, 100%: 0.994

## Conformal sets (alpha = 0.10)

- A rho=0.0: mixture coverage full 0.891 (size 1.16), live 0.899 (size 1.91); on residue devices full 0.378, live 1.000.
- C rho=0.0: mixture coverage full 0.898 (size 1.10), live 0.896 (size 1.95); on residue devices full 0.262, live 1.000.
- C rho=0.25: mixture coverage full 0.896 (size 1.14), live 0.899 (size 1.90); on residue devices full 0.531, live 0.998.
- Own LAC sets identical to MAPIE for 100.0% of cases (worst model).

## Alignment with N4 and N2

- FedAvg on N4's partition (alpha 0.5), mean macro-F1 over levels: B federated 0.827 vs centralised 0.837; C federated 0.828 vs centralised 0.833; L federated 0.714 vs centralised 0.727
- CASE/UCO export with RecoveredObjectFacet + ConfidenceFacet: 4/4 files valid (CASE 1.5.0); features from CASE equal step 6 (max diff 1.4e-14).
- N2 DP release of recovered reliance, mean abs error by epsilon: 0.5: 0.0191, 1: 0.0126, 2: 0.0058, 4: 0.0029, 8: 0.0016, inf: 0.0000

## Side by side: N5 (teammate's N3) vs N5b (alternative N3), same phones

| measure | N5 | N5b |
|---|---|---|
| deleted planted evidence recovered | 0.398 | 0.417 |
| deleted background recovered | 0.668 | 0.690 |
| foreign residue recovered | 0.459 | 0.479 |
| recovered records that are 'wrong' | 0.0018 | 0.0148 |
| ECE of p, raw (test) | 0.3403 | 0.0029 |
| ECE of p, after isotonic (test) | 0.0010 | 0.0006 |
| macro-F1 L, deleted 0% (rho 0, mean of seeds) | 0.923 | 0.923 |
| macro-F1 L, deleted 10% (rho 0, mean of seeds) | 0.914 | 0.914 |
| macro-F1 L, deleted 25% (rho 0, mean of seeds) | 0.887 | 0.887 |
| macro-F1 L, deleted 50% (rho 0, mean of seeds) | 0.801 | 0.801 |
| macro-F1 L, deleted 100% (rho 0, mean of seeds) | 0.107 | 0.107 |
| macro-F1 A, deleted 0% (rho 0, mean of seeds) | 0.922 | 0.919 |
| macro-F1 A, deleted 10% (rho 0, mean of seeds) | 0.914 | 0.910 |
| macro-F1 A, deleted 25% (rho 0, mean of seeds) | 0.899 | 0.894 |
| macro-F1 A, deleted 50% (rho 0, mean of seeds) | 0.848 | 0.850 |
| macro-F1 A, deleted 100% (rho 0, mean of seeds) | 0.594 | 0.619 |
| macro-F1 Af, deleted 0% (rho 0, mean of seeds) | 0.919 | 0.918 |
| macro-F1 Af, deleted 10% (rho 0, mean of seeds) | 0.913 | 0.911 |
| macro-F1 Af, deleted 25% (rho 0, mean of seeds) | 0.894 | 0.898 |
| macro-F1 Af, deleted 50% (rho 0, mean of seeds) | 0.840 | 0.845 |
| macro-F1 Af, deleted 100% (rho 0, mean of seeds) | 0.594 | 0.621 |
| macro-F1 B, deleted 0% (rho 0, mean of seeds) | 0.920 | 0.919 |
| macro-F1 B, deleted 10% (rho 0, mean of seeds) | 0.912 | 0.911 |
| macro-F1 B, deleted 25% (rho 0, mean of seeds) | 0.897 | 0.894 |
| macro-F1 B, deleted 50% (rho 0, mean of seeds) | 0.847 | 0.850 |
| macro-F1 B, deleted 100% (rho 0, mean of seeds) | 0.590 | 0.622 |
| macro-F1 C, deleted 0% (rho 0, mean of seeds) | 0.892 | 0.891 |
| macro-F1 C, deleted 10% (rho 0, mean of seeds) | 0.890 | 0.890 |
| macro-F1 C, deleted 25% (rho 0, mean of seeds) | 0.879 | 0.881 |
| macro-F1 C, deleted 50% (rho 0, mean of seeds) | 0.858 | 0.861 |
| macro-F1 C, deleted 100% (rho 0, mean of seeds) | 0.772 | 0.781 |
| innocent residue devices triaged as offence, L rho 0.0 | 0.202 | 0.202 |
| innocent residue devices triaged as offence, L rho 0.25 | 0.202 | 0.202 |
| innocent residue devices triaged as offence, A rho 0.0 | 0.662 | 0.691 |
| innocent residue devices triaged as offence, A rho 0.25 | 0.642 | 0.667 |
| innocent residue devices triaged as offence, B rho 0.0 | 0.664 | 0.691 |
| innocent residue devices triaged as offence, B rho 0.25 | 0.644 | 0.671 |
| innocent residue devices triaged as offence, C rho 0.0 | 0.747 | 0.784 |
| innocent residue devices triaged as offence, C rho 0.25 | 0.633 | 0.642 |
| innocent residue devices triaged as offence, A rho N1 | 0.540 | 0.560 |
