# N1 results summary (all numbers from this run)

Seed: 2026. Test split: 600 cases.

## Triage models (test macro-F1)

- MLP: 0.925; XGBoost: 0.941
- MLP accuracy by tier: easy 0.985, mixed 0.950, subtle 0.840

## Oracle check

- MLP macro-F1: all features 0.925; without decisive 0.828; without decisive and proxies 0.217; only decisive 0.926
- Proxy features: sms_out_total, sms_unique_contacts, call_out_total, call_mean_duration, web_total_visits, web_social_share, web_distinct_domains, login_total, login_failed_total, txn_in_count, night_activity_share
- MLP per-case pass rate: 0.940 (easy 1.000, mixed 1.000, subtle 0.820)
- XGB per-case pass rate: 0.964 (easy 1.000, mixed 1.000, subtle 0.893)

## Evidence recall@|G| (oracle-passed cases, mean [95% CI])

| Explainer | easy | mixed | subtle | all |
|---|---|---|---|---|
| KernelSHAP | 0.889 [0.86, 0.91] | 0.889 [0.86, 0.91] | 0.878 [0.84, 0.91] | 0.886 [0.87, 0.90] |
| Integrated Gradients | 0.889 [0.86, 0.91] | 0.889 [0.86, 0.91] | 0.825 [0.78, 0.87] | 0.870 [0.85, 0.89] |
| GradientSHAP | 0.889 [0.86, 0.91] | 0.887 [0.86, 0.91] | 0.870 [0.83, 0.91] | 0.883 [0.86, 0.90] |
| LIME | 0.889 [0.86, 0.91] | 0.889 [0.86, 0.91] | 0.817 [0.77, 0.86] | 0.868 [0.85, 0.89] |
| Feature Ablation | 0.889 [0.86, 0.91] | 0.887 [0.86, 0.91] | 0.886 [0.85, 0.92] | 0.887 [0.87, 0.90] |
| Random (floor) | 0.111 [0.08, 0.14] | 0.098 [0.07, 0.12] | 0.093 [0.06, 0.13] | 0.101 [0.09, 0.12] |
| TreeSHAP (XGBoost ref.) | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 0.806 [0.76, 0.85] | 0.940 [0.92, 0.96] |

## F-Fidelity (pooled, oracle-passed cases)

| Explainer | FFid+ | FFid- | FFid+ (prob) | FFid- (prob) | naive deletion drop |
|---|---|---|---|---|---|
| KernelSHAP | 0.126 [0.105, 0.147] | 0.002 | 0.125 | -0.002 | 0.846 |
| Integrated Gradients | 0.084 [0.070, 0.099] | 0.003 | 0.090 | 0.001 | 0.795 |
| GradientSHAP | 0.111 [0.093, 0.130] | 0.002 | 0.111 | -0.001 | 0.794 |
| LIME | 0.091 [0.074, 0.107] | 0.002 | 0.093 | 0.001 | 0.762 |
| Feature Ablation | 0.111 [0.092, 0.131] | 0.001 | 0.116 | -0.002 | 0.842 |
| Random (floor) | 0.018 [0.010, 0.026] | 0.019 | 0.014 | 0.016 | 0.043 |

## Acceptance test

Level 1 (method validation): KernelSHAP VALIDATED, Integrated Gradients VALIDATED, GradientSHAP VALIDATED, LIME VALIDATED, Feature Ablation VALIDATED, Random (floor) not validated

Level 2 confidence cut-off calibrated on the validation split: tau = 0.95

| Subset (test) | n | share good | validated | P(good \| validated) | bad caught | good rejected |
|---|---|---|---|---|---|---|
| all offence cases | 450 | 0.878 | 0.731 | 0.964 | 0.782 | 0.197 |
| oracle-passed | 423 | 0.929 | 0.771 | 0.969 | 0.667 | 0.196 |
| oracle-failed | 27 | 0.074 | 0.111 | 0.333 | 0.920 | 0.500 |
| tier=easy | 150 | 1.000 | 1.000 | 1.000 | n/a | 0.000 |
| tier=mixed | 150 | 1.000 | 1.000 | 1.000 | n/a | 0.000 |
| tier=subtle | 150 | 0.633 | 0.193 | 0.586 | 0.782 | 0.821 |

## Scaffold attack (optional step)

- KernelSHAP: recall 0.889 -> 0.679; innocent feature ranked first in 0.320 of cases
- LIME: recall 0.872 -> 0.084; innocent feature ranked first in 1.000 of cases
- Integrated Gradients: recall 0.872 -> 0.607; innocent feature ranked first in 0.800 of cases
- Feature Ablation: recall 0.890 -> 0.492; innocent feature ranked first in 0.047 of cases
- Agreement check flags 0.100 of honest-model cases and 0.173 of scaffold cases; fooled primary explanations caught: 0.256
