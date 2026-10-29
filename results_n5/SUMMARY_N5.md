# UniForensic N5 - summary of this run

## N3 (teammate's code, unchanged) on N1's databases

- Teammate classifier reproduced bit for bit: True (numpy seed 2026).
- Deleted planted evidence recovered: 39.8% (background 66.8%, foreign residue 45.9%).
- Recovered records that are fabrications ('wrong'): 0.18%.
- ECE of N3's p: in-sample (N3's report) 0.0236; on N1 test databases 0.3403; after isotonic recalibration 0.0010.
- Correctly recovered deleted records the fuzzy live filter would discard: 63.3% (planted evidence: 19.0%).

## Triage (macro-F1, mean +- SD over 3 seeds, rho = 0, clean benign devices)

| variant | deleted 0% | deleted 10% | deleted 25% | deleted 50% | deleted 100% |
|---|---|---|---|---|---|
| L | 0.923 +- 0.004 | 0.914 +- 0.007 | 0.887 +- 0.002 | 0.801 +- 0.009 | 0.107 +- 0.009 |
| A | 0.922 +- 0.005 | 0.914 +- 0.004 | 0.899 +- 0.005 | 0.848 +- 0.005 | 0.594 +- 0.002 |
| Af | 0.919 +- 0.005 | 0.913 +- 0.005 | 0.894 +- 0.004 | 0.840 +- 0.004 | 0.594 +- 0.001 |
| B | 0.920 +- 0.005 | 0.912 +- 0.005 | 0.897 +- 0.004 | 0.847 +- 0.007 | 0.590 +- 0.002 |
| C | 0.892 +- 0.004 | 0.890 +- 0.003 | 0.879 +- 0.003 | 0.858 +- 0.007 | 0.772 +- 0.007 |
| A, N1 model unchanged | 0.929 | 0.916 | 0.888 | 0.838 | 0.576 |

## Explanations (TER, rho = 0)

| variant | 0% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| L | 0.830 | 0.789 | 0.736 | 0.662 | 0.002 |
| A | 0.850 | 0.825 | 0.790 | 0.731 | 0.451 |
| B | 0.849 | 0.823 | 0.789 | 0.728 | 0.449 |
| C | 0.807 | 0.802 | 0.791 | 0.773 | 0.842 |

## Innocent devices triaged as offences

| variant | rho | clean | with foreign residue |
|---|---|---|---|
| L | 0.0 | 0.202 | 0.202 |
| L | 0.25 | 0.202 | 0.202 |
| A | 0.0 | 0.244 | 0.662 |
| A | 0.25 | 0.240 | 0.642 |
| B | 0.0 | 0.251 | 0.664 |
| B | 0.25 | 0.249 | 0.644 |
| C | 0.0 | 0.231 | 0.747 |
| C | 0.25 | 0.256 | 0.633 |
| A | N1 | 0.113 | 0.540 |

## Recovery-dependence flag

- Residue false incriminations caused by recovered records (C, rho 0): 82.4%; all carry the counterfactual flag (exact test).
- Attribution flag (C) vs counterfactual: agreement 0.869, catches 80.5% of recovery-caused false incriminations.
- Workload (correct offence decisions flagged, C rho 0): 0%: 0.012, 10%: 0.030, 25%: 0.076, 50%: 0.168, 100%: 0.993

## Conformal sets (alpha = 0.10)

- A rho=0.0: mixture coverage full 0.892 (size 1.17), live 0.908 (size 1.96); on residue devices full 0.396, live 1.000.
- C rho=0.0: mixture coverage full 0.898 (size 1.10), live 0.909 (size 2.01); on residue devices full 0.293, live 1.000.
- C rho=0.25: mixture coverage full 0.893 (size 1.13), live 0.904 (size 1.90); on residue devices full 0.544, live 0.998.
- Own LAC sets identical to MAPIE for 100.0% of cases (worst model).

## Alignment with N4 and N2

- FedAvg on N4's partition (alpha 0.5), mean macro-F1 over levels: B federated 0.821 vs centralised 0.834; C federated 0.820 vs centralised 0.830; L federated 0.714 vs centralised 0.727
- CASE/UCO export with RecoveredObjectFacet + ConfidenceFacet: 4/4 files valid (CASE 1.5.0); features from CASE equal step 6 (max diff 1.4e-14).
- N2 DP release of recovered reliance, mean abs error by epsilon: 0.5: 0.0196, 1: 0.0128, 2: 0.0059, 4: 0.0030, 8: 0.0016, inf: 0.0000
