# Example N5b case reports (alternative N3; variant C, rho = 0.25, seed 2026, test split)

Generated automatically from the run. 'Truth' lines are what the planted answer key says;
an examiner's report would not have them.

## Offence, all planted evidence deleted

Case version `TE00150|1|0`

- Triage: **fraud** (probability 0.84).
- 90% prediction set with recovered records: {fraud}.
- 90% prediction set from live data only: {benign}.
- Without the recovered records the model says: benign.
- Recovered records: 58 (carved from deleted space; calibrated confidence 1.00-1.00); recovered channel of the decisive features: sms_scam_kw_out +9.0, txn_in_new_counterparty +20.0, sms_max_recipients_1h +14.0.
- Share of the decisive-feature attribution on recovered records: 80%.
- **This conclusion depends on recovered records.** Their confidence says the records are intact, not who created them: confirm attribution (who used the device, when) before relying on them.
- Truth: fraud; recovered records by origin: correct 58

## Innocent device holding foreign residue

Case version `TE00000|0|1`

- Triage: **harassment** (probability 0.78).
- 90% prediction set with recovered records: {harassment}.
- 90% prediction set from live data only: {benign}.
- Without the recovered records the model says: benign.
- Recovered records: 54 (carved from deleted space; calibrated confidence 1.00-1.00); recovered channel of the decisive features: call_max_short_out_single_number +14.0, call_out_night_unanswered +10.0.
- Share of the decisive-feature attribution on recovered records: 73%.
- **This conclusion depends on recovered records.** Their confidence says the records are intact, not who created them: confirm attribution (who used the device, when) before relying on them.
- Truth: benign; recovered records by origin: correct 28, foreign 26

## Offence, nothing of the evidence deleted

Case version `TE00150|0|0`

- Triage: **fraud** (probability 1.00).
- 90% prediction set with recovered records: {fraud}.
- 90% prediction set from live data only: {fraud}.
- Without the recovered records the model says: fraud.
- Recovered records: 14 (carved from deleted space; calibrated confidence 1.00-1.00); recovered channel of the decisive features: none.
- Share of the decisive-feature attribution on recovered records: 0%.
- The conclusion does not depend on recovered records: live data alone gives the same class.
- Truth: fraud; recovered records by origin: correct 14
