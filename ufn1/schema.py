"""The unified evidence schema and the answer-key vocabulary.

Every artifact, whatever device it came from, is stored as ONE record type:

    case_id, artifact_id, artifact_type, device, source, timestamp,
    role, pattern_id, + type-specific columns

* artifact_type : sms | call | web | login | txn
* device        : android | windows   (kept for the later CASE/UCO mapping, N4)
* source        : live | recovered    (all 'live' in N1; N5 will use 'recovered')
* role          : background | evidence | distractor
                  'evidence'   = planted decisive artifact (this is the answer key)
                  'distractor' = planted look-alike from ANOTHER class (not evidence)
* pattern_id    : which planted pattern produced it (empty for background)
"""

CLASSES = ["benign", "fraud", "harassment", "cybercrime"]

# ---------------------------------------------------------------------
# Planted decisive patterns: 3 per offence class.
# Each pattern is designed to drive exactly ONE decisive feature.
# ---------------------------------------------------------------------
PATTERNS = {
    "fraud": {
        "F1_scam_sms":       "sms_scam_kw_out",           # scam texts (KYC / OTP / prize)
        "F2_mule_inflow":    "txn_in_new_counterparty",   # money from many new payers
        "F3_sms_blast":      "sms_max_recipients_1h",     # one-hour blast to many numbers
    },
    "harassment": {
        "H1_abusive_sms":    "sms_abusive_kw_out",        # threatening / abusive texts
        "H2_repeat_calls":   "call_max_short_out_single_number",# many calls to one victim
        "H3_night_calls":    "call_out_night_unanswered", # unanswered calls 00:00-05:00
    },
    "cybercrime": {
        "C1_hacktool_web":   "web_hacktool_visits",       # exploit / booter sites
        "C2_cred_stuffing":  "login_failed_distinct_accounts",  # many accounts tried
        "C3_vpn_hopping":    "login_distinct_ip_countries",     # logins from many countries
    },
}

DECISIVE_FEATURES = [f for pats in PATTERNS.values() for f in pats.values()]

# Background (non-decisive) features. Some of them are *mechanically* touched by
# planted artifacts (e.g. sms_out_total grows when scam SMS are planted); the
# feature step detects this per case and stores it as the "affected" set.
BACKGROUND_FEATURES = [
    "sms_in_total", "sms_out_total", "sms_unique_contacts", "sms_promo_in",
    "sms_bank_otp_in", "call_in_total", "call_out_total", "call_mean_duration",
    "call_unanswered_in", "web_total_visits", "web_social_share",
    "web_distinct_domains", "login_total", "login_failed_total",
    "txn_out_count", "txn_out_amount_log", "txn_in_count", "night_activity_share",
]

FEATURES = DECISIVE_FEATURES + BACKGROUND_FEATURES   # model input order (27)

# Features that are counts -> log1p before scaling (heavy-tailed)
COUNT_FEATURES = [f for f in FEATURES
                  if f not in ("call_mean_duration", "web_social_share",
                               "night_activity_share", "txn_out_amount_log")]

PATTERN_TO_FEATURE = {p: f for pats in PATTERNS.values() for p, f in pats.items()}
PATTERN_TO_CLASS = {p: c for c, pats in PATTERNS.items() for p in pats}

ARTIFACT_TYPES = ["sms", "call", "web", "login", "txn"]
