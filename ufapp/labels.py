"""Plain-language names for the model's inputs (shown in explanations and reports).
The feature definitions themselves live in ufn1/features.py; these are only labels."""
from __future__ import annotations

from ufn1.schema import DECISIVE_FEATURES, FEATURES, PATTERN_TO_CLASS, PATTERN_TO_FEATURE

FEATURE_LABEL = {
    "sms_scam_kw_out": "Outgoing SMS with scam keywords (KYC, OTP, prize)",
    "txn_in_new_counterparty": "Incoming payments from first-time payers",
    "sms_max_recipients_1h": "Most different numbers texted within one hour",
    "sms_abusive_kw_out": "Outgoing SMS with threatening or abusive words",
    "call_max_short_out_single_number": "Most short or unanswered calls to one number",
    "call_out_night_unanswered": "Unanswered outgoing calls between 00:00 and 05:00",
    "web_hacktool_visits": "Visits to blocklisted hacking-tool sites",
    "login_failed_distinct_accounts": "Different accounts in failed logins",
    "login_distinct_ip_countries": "Different countries of login IP addresses",
    "sms_in_total": "Incoming SMS",
    "sms_out_total": "Outgoing SMS",
    "sms_unique_contacts": "Different phone numbers in SMS",
    "sms_promo_in": "Incoming promotional SMS",
    "sms_bank_otp_in": "Incoming bank / OTP SMS",
    "call_in_total": "Incoming calls",
    "call_out_total": "Outgoing calls",
    "call_mean_duration": "Mean duration of answered calls (s)",
    "call_unanswered_in": "Missed incoming calls",
    "web_total_visits": "Web visits",
    "web_social_share": "Share of web visits to social media",
    "web_distinct_domains": "Different web domains",
    "login_total": "Logins",
    "login_failed_total": "Failed logins",
    "txn_out_count": "Outgoing payments",
    "txn_out_amount_log": "Amount paid out (log)",
    "txn_in_count": "Incoming payments",
    "night_activity_share": "Share of activity between 00:00 and 05:00",
}
assert set(FEATURE_LABEL) == set(FEATURES)

FEATURE_CLASS = {f: PATTERN_TO_CLASS[p] for p, f in PATTERN_TO_FEATURE.items()}  # decisive -> offence

TABLE_LABEL = {"sms": "Messages (messages)", "call": "Call log (calls)",
               "web": "Browser history (browser_history)", "login": "Login log (logins)",
               "txn": "Payments (transactions)"}


def input_names(variant: str) -> list[str]:
    """Names of the model's inputs in order (27, or 36 for variant C)."""
    if variant != "C":
        return list(FEATURES)
    return list(FEATURES) + [f"rec_{f}" for f in DECISIVE_FEATURES]


def input_label(name: str) -> str:
    if name.startswith("rec_"):
        return "RECOVERED: " + FEATURE_LABEL[name[4:]]
    return FEATURE_LABEL[name]
