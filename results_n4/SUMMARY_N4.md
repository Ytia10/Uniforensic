# N4 results summary (all numbers from this run)

Seeds: [2026, 2027, 2028]. 8 agencies. Test set = N1's 600 cases. Explainer for TER/recall/proxy: integrated_gradients.

## CASE/UCO check

- 60 cases exported (32601 CASE objects); feature mismatches after round trip: 0
- Official validation (case-1.5.0): TE00000 conforms=True, TE00001 conforms=True

## Macro-F1 by alpha (mean +/- SD over seeds)

| alpha | Centralised | FedAvg | FedProx | Local-only mean [min, max] |
|---|---|---|---|---|
| 0.1 | 0.930 +/- 0.006 | 0.906 +/- 0.024 | 0.919 +/- 0.013 | 0.364 +/- 0.051 [0.100, 0.825] |
| 0.5 | 0.930 +/- 0.006 | 0.926 +/- 0.006 | 0.920 +/- 0.003 | 0.609 +/- 0.081 [0.322, 0.918] |
| 1 | 0.930 +/- 0.006 | 0.926 +/- 0.002 | 0.924 +/- 0.011 | 0.796 +/- 0.060 [0.360, 0.899] |
| 100 | 0.930 +/- 0.006 | 0.932 +/- 0.004 | 0.932 +/- 0.004 | 0.898 +/- 0.005 [0.879, 0.916] |

## TER by alpha (mean +/- SD over seeds)

| alpha | Centralised | FedAvg | FedProx | Local-only mean [min, max] |
|---|---|---|---|---|
| 0.1 | 0.860 +/- 0.010 | 0.789 +/- 0.079 | 0.825 +/- 0.015 | 0.371 +/- 0.042 [0.000, 0.809] |
| 0.5 | 0.860 +/- 0.010 | 0.848 +/- 0.015 | 0.839 +/- 0.012 | 0.562 +/- 0.061 [0.227, 0.787] |
| 1 | 0.860 +/- 0.010 | 0.835 +/- 0.008 | 0.841 +/- 0.022 | 0.712 +/- 0.035 [0.484, 0.800] |
| 100 | 0.860 +/- 0.010 | 0.852 +/- 0.013 | 0.856 +/- 0.020 | 0.764 +/- 0.006 [0.720, 0.820] |

## Evidence recall@|G| (oracle-passed) by alpha (mean +/- SD over seeds)

| alpha | Centralised | FedAvg | FedProx | Local-only mean [min, max] |
|---|---|---|---|---|
| 0.1 | 0.886 +/- 0.036 | 0.831 +/- 0.034 | 0.861 +/- 0.021 | 0.781 +/- 0.001 [0.612, 0.949] |
| 0.5 | 0.886 +/- 0.036 | 0.932 +/- 0.021 | 0.889 +/- 0.036 | 0.758 +/- 0.030 [0.627, 0.931] |
| 1 | 0.886 +/- 0.036 | 0.878 +/- 0.024 | 0.929 +/- 0.013 | 0.790 +/- 0.009 [0.702, 0.897] |
| 100 | 0.886 +/- 0.036 | 0.916 +/- 0.043 | 0.937 +/- 0.020 | 0.805 +/- 0.012 [0.725, 0.895] |

## Oracle pass rate by alpha (mean +/- SD over seeds)

| alpha | Centralised | FedAvg | FedProx | Local-only mean [min, max] |
|---|---|---|---|---|
| 0.1 | 0.946 +/- 0.006 | 0.881 +/- 0.140 | 0.924 +/- 0.058 | 0.347 +/- 0.041 [0.000, 0.933] |
| 0.5 | 0.946 +/- 0.006 | 0.932 +/- 0.003 | 0.933 +/- 0.008 | 0.602 +/- 0.069 [0.280, 0.911] |
| 1 | 0.946 +/- 0.006 | 0.941 +/- 0.001 | 0.936 +/- 0.013 | 0.797 +/- 0.060 [0.378, 0.927] |
| 100 | 0.946 +/- 0.006 | 0.946 +/- 0.006 | 0.947 +/- 0.007 | 0.886 +/- 0.005 [0.822, 0.922] |

## Local-only models: crime types rarely seen at the agency

| Cases of that type at the agency | test cases | models | detected | TER | proxy share |
|---|---|---|---|---|---|
| 0 | 4800 | 24 | 0.000 | 0.000 | 0.128 |
| 1-9 | 4500 | 26 | 0.272 | 0.228 | 0.129 |
| 10-29 | 6000 | 33 | 0.831 | 0.697 | 0.154 |
| 30-99 | 21300 | 71 | 0.923 | 0.748 | 0.153 |
| 100+ | 6600 | 35 | 0.974 | 0.738 | 0.148 |

FedAvg (all alphas, all seeds): detected 0.955, TER 0.831

## Proxy share (correctly classified offence cases)

| Setting | mean | SD | models |
|---|---|---|---|
| Centralised | 0.071 | 0.038 | 3 |
| FedAvg | 0.070 | 0.051 | 12 |
| FedProx | 0.054 | 0.036 | 12 |
| Local-only | 0.164 | 0.070 | 96 |

## Poisoning (alpha 0.5, mean +/- SD over seeds)

Attacker per seed: seed 2026: agency 0 (30% of cases); seed 2027: agency 5 (20% of cases); seed 2028: agency 0 (9% of cases)

| Rule | Attack | Macro-F1 | Fraud F1 | TER |
|---|---|---|---|---|
| FedAvg | no attack | 0.926 +/- 0.004 | 0.942 +/- 0.004 | 0.841 +/- 0.008 |
| FedAvg | label flip (fraud -> benign) | 0.825 +/- 0.163 | 0.632 +/- 0.503 | 0.749 +/- 0.139 |
| FedAvg | scaled update (x-5) | 0.433 +/- 0.401 | 0.279 +/- 0.484 | 0.329 +/- 0.400 |
| FedMedian | no attack | 0.922 +/- 0.005 | 0.933 +/- 0.023 | 0.836 +/- 0.014 |
| FedMedian | label flip (fraud -> benign) | 0.910 +/- 0.019 | 0.906 +/- 0.049 | 0.824 +/- 0.007 |
| FedMedian | scaled update (x-5) | 0.898 +/- 0.020 | 0.892 +/- 0.057 | 0.815 +/- 0.014 |
| FedTrimmedAvg | no attack | 0.929 +/- 0.005 | 0.943 +/- 0.013 | 0.839 +/- 0.007 |
| FedTrimmedAvg | label flip (fraud -> benign) | 0.908 +/- 0.018 | 0.904 +/- 0.047 | 0.825 +/- 0.025 |
| FedTrimmedAvg | scaled update (x-5) | 0.904 +/- 0.026 | 0.896 +/- 0.053 | 0.819 +/- 0.034 |
| Krum | no attack | 0.623 +/- 0.087 | 0.672 +/- 0.171 | 0.590 +/- 0.211 |
| Krum | label flip (fraud -> benign) | 0.540 +/- 0.036 | 0.487 +/- 0.381 | 0.502 +/- 0.125 |
| Krum | scaled update (x-5) | 0.602 +/- 0.078 | 0.623 +/- 0.193 | 0.487 +/- 0.213 |

## Convergence (FedAvg, validation macro-F1)

| alpha | round reaching 90% of best | final |
|---|---|---|
| 0.1 | 16 | 0.899 |
| 0.5 | 5 | 0.926 |
| 1 | 5 | 0.932 |
| 100 | 5 | 0.932 |
