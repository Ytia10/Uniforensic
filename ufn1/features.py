"""Turn unified-schema artifacts into ONE fixed case-level feature vector.

This vector is what the triage model sees. Because every agency can compute the
same vector from the same schema, it is also the shared input space for the
federated setting (N4) later.

The keyword lists below play the role of a simple forensic text classifier.
They are applied to the text, not to the generator's internal labels, so
background noise ('My insta account got blocked lol') can trigger them too.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import COUNT_FEATURES, FEATURES

SCAM_RE = re.compile(r"\b(?:kyc|share otp|blocked|won a prize|verify your account|deactivated)\b", re.I)
ABUSIVE_RE = re.compile(r"(?:i know where you live|you will regret|answer my calls|watch your back|worthless)", re.I)
BANK_RE = re.compile(r"\b(?:otp|debited|credited)\b", re.I)
SOCIAL_DOMAINS = {"socialbook.example", "chatgram.example", "vidstream.example"}
# a "threat-intelligence blocklist" of known hacking-tool sites
HACKTOOL_BLOCKLIST = {"exploit-archive.example", "booter-stress.example", "credleaks.example",
                      "rat-builder.example", "carding-forum.example"}
NIGHT_HOURS = 5          # 00:00-04:59 counts as night


def _ts(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], format="ISO8601")


def extract_features(tables: dict[str, pd.DataFrame], case_ids) -> pd.DataFrame:
    """tables: {'sms': df, 'call': df, 'web': df, 'login': df, 'txn': df}.
    Returns a DataFrame (index = case_id) with the 27 FEATURES columns (raw scale)."""
    out = pd.DataFrame(index=pd.Index(case_ids, name="case_id"))

    def put(name, series):
        out[name] = series.reindex(out.index).fillna(0.0).astype(float)

    # ------------------------------------------------------------------ SMS
    s = tables["sms"]
    s_out, s_in = s[s.direction == "out"], s[s.direction == "in"]
    alpha_sender = ~s_in.counterparty.astype(str).str.startswith("+")
    bank = s_in.text.str.contains(BANK_RE)
    put("sms_in_total", s_in.groupby("case_id").size())
    put("sms_out_total", s_out.groupby("case_id").size())
    put("sms_unique_contacts", s[s.counterparty.astype(str).str.startswith("+")]
        .groupby("case_id").counterparty.nunique())
    put("sms_promo_in", s_in[alpha_sender & ~bank].groupby("case_id").size())
    put("sms_bank_otp_in", s_in[alpha_sender & bank].groupby("case_id").size())
    put("sms_scam_kw_out", s_out[s_out.text.str.contains(SCAM_RE)].groupby("case_id").size())
    put("sms_abusive_kw_out", s_out[s_out.text.str.contains(ABUSIVE_RE)].groupby("case_id").size())
    so = s_out.assign(hour_bucket=_ts(s_out).dt.floor("h"))
    put("sms_max_recipients_1h", so.groupby(["case_id", "hour_bucket"]).counterparty.nunique()
        .groupby("case_id").max())

    # ---------------------------------------------------------------- calls
    c = tables["call"]
    c_out, c_in = c[c.direction == "out"], c[c.direction == "in"]
    answered = c.answered.astype(str).str.lower() == "true"
    c_hour = _ts(c).dt.hour
    put("call_in_total", c_in.groupby("case_id").size())
    put("call_out_total", c_out.groupby("case_id").size())
    put("call_mean_duration", c[answered].groupby("case_id").duration_s.mean())
    put("call_unanswered_in", c[(c.direction == "in") & ~answered].groupby("case_id").size())
    # repeated SHORT (<30 s) or unanswered outgoing calls to one number
    short = c_out[(c_out.duration_s < 30)]
    put("call_max_short_out_single_number", short.groupby(["case_id", "counterparty"]).size()
        .groupby("case_id").max())
    put("call_out_night_unanswered",
        c[(c.direction == "out") & ~answered & (c_hour < NIGHT_HOURS)].groupby("case_id").size())

    # ------------------------------------------------------------------ web
    w = tables["web"]
    put("web_total_visits", w.groupby("case_id").size())
    put("web_distinct_domains", w.groupby("case_id").domain.nunique())
    put("web_social_share", w.domain.isin(SOCIAL_DOMAINS).groupby(w.case_id).mean())
    put("web_hacktool_visits", w[w.domain.isin(HACKTOOL_BLOCKLIST)].groupby("case_id").size())

    # --------------------------------------------------------------- logins
    lg = tables["login"]
    ok = lg.success.astype(str).str.lower() == "true"
    put("login_total", lg.groupby("case_id").size())
    put("login_failed_total", lg[~ok].groupby("case_id").size())
    put("login_failed_distinct_accounts", lg[~ok].groupby("case_id").account.nunique())
    put("login_distinct_ip_countries", lg.groupby("case_id").ip_country.nunique())

    # --------------------------------------------------------- transactions
    t = tables["txn"]
    t_out, t_in = t[t.direction == "out"], t[t.direction == "in"]
    put("txn_out_count", t_out.groupby("case_id").size())
    put("txn_out_amount_log", np.log1p(t_out.groupby("case_id").amount.sum()))
    put("txn_in_count", t_in.groupby("case_id").size())
    seen = t.groupby(["case_id", "counterparty"]).size().rename("n_seen")
    t_in2 = t_in.join(seen, on=["case_id", "counterparty"])
    put("txn_in_new_counterparty", t_in2[t_in2.n_seen == 1].groupby("case_id").size())

    # ------------------------------------------------- cross-artifact feature
    hours = pd.concat([pd.DataFrame({"case_id": d.case_id, "h": _ts(d).dt.hour})
                       for d in tables.values() if len(d)])
    put("night_activity_share", (hours.h < NIGHT_HOURS).groupby(hours.case_id).mean())

    return out[FEATURES]


# ---------------------------------------------------------------- scaling
def log_transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df[COUNT_FEATURES] = np.log1p(df[COUNT_FEATURES])
    return df


def fit_scaler(df_raw_train: pd.DataFrame) -> dict:
    x = log_transform(df_raw_train)[FEATURES].to_numpy()
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    return {"features": FEATURES, "mean": mean.tolist(), "scale": std.tolist()}


def apply_scaler(df_raw: pd.DataFrame, scaler: dict) -> np.ndarray:
    """Raw features -> model input (log1p on counts, then z-score).
    In this space, 0 = the training mean -> a natural 'feature removed' baseline."""
    x = log_transform(df_raw)[FEATURES].to_numpy(dtype=np.float64)
    return ((x - np.array(scaler["mean"])) / np.array(scaler["scale"])).astype(np.float32)
