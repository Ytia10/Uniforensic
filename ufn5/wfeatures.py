"""N1's 27 case features, computed from artifacts that each carry a weight w in [0, 1].

  live artifact          w = 1
  recovered artifact     w = its confidence (variant B), or 1 (variant A)

Each feature is the EXPECTED value of N1's feature if every recovered record exists
independently with probability w (this is the "uncertain data" view of Reis et al.,
Probabilistic Random Forest, 2019, applied here to aggregated case features):
  counts                         sum of w
  distinct values (nunique)      sum over values of P(at least one record) = 1 - prod(1 - w)
  'seen exactly once'            w_r * prod over the other records of (1 - w)
  means / shares                 weighted mean (plug-in)
  maxima                         max over groups of the expected group value (plug-in)
With every w = 1 these formulas ARE N1's features: n5_00 checks equality with
ufn1.features.extract_features on all 3,600 N1 cases (to 1e-9).
The keyword lists, blocklists and night hours are imported from N1, not copied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ufn1.features import (ABUSIVE_RE, BANK_RE, HACKTOOL_BLOCKLIST, NIGHT_HOURS, SCAM_RE,
                           SOCIAL_DOMAINS, _ts)
from ufn1.schema import FEATURES


def _p_any(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Per group in keys: P(at least one record exists) = 1 - prod(1 - w)."""
    if not len(df):
        return pd.Series(dtype=float)
    q = (1.0 - df["w"]).clip(lower=0.0)
    return 1.0 - q.groupby([df[k] for k in keys]).prod()


def _sum_w(df: pd.DataFrame) -> pd.Series:
    return df["w"].groupby(df["case_id"]).sum()


def extract_features_weighted(tables: dict[str, pd.DataFrame], case_ids) -> pd.DataFrame:
    """tables: {'sms','call','web','login','txn'} -> DataFrames with N1's columns + 'w'.
    Returns a DataFrame (index = case_id) with the 27 FEATURES (raw scale)."""
    out = pd.DataFrame(index=pd.Index(case_ids, name="case_id"))

    def put(name, series):
        out[name] = series.reindex(out.index).fillna(0.0).astype(float)

    # ------------------------------------------------------------------ SMS
    s = tables["sms"]
    s_out, s_in = s[s.direction == "out"], s[s.direction == "in"]
    alpha_sender = ~s_in.counterparty.astype(str).str.startswith("+")
    bank = s_in.text.str.contains(BANK_RE)
    put("sms_in_total", _sum_w(s_in))
    put("sms_out_total", _sum_w(s_out))
    phone = s[s.counterparty.astype(str).str.startswith("+")]
    put("sms_unique_contacts", _p_any(phone, ["case_id", "counterparty"]).groupby(level=0).sum())
    put("sms_promo_in", _sum_w(s_in[alpha_sender & ~bank]))
    put("sms_bank_otp_in", _sum_w(s_in[alpha_sender & bank]))
    put("sms_scam_kw_out", _sum_w(s_out[s_out.text.str.contains(SCAM_RE)]))
    put("sms_abusive_kw_out", _sum_w(s_out[s_out.text.str.contains(ABUSIVE_RE)]))
    so = s_out.assign(hour_bucket=_ts(s_out).dt.floor("h"))
    put("sms_max_recipients_1h", _p_any(so, ["case_id", "hour_bucket", "counterparty"])
        .groupby(level=[0, 1]).sum().groupby(level=0).max())

    # ---------------------------------------------------------------- calls
    c = tables["call"]
    c_out, c_in = c[c.direction == "out"], c[c.direction == "in"]
    answered = c.answered.astype(str).str.lower() == "true"
    c_hour = _ts(c).dt.hour
    put("call_in_total", _sum_w(c_in))
    put("call_out_total", _sum_w(c_out))
    ca = c[answered]
    put("call_mean_duration", (ca.w * ca.duration_s).groupby(ca.case_id).sum() / _sum_w(ca))
    put("call_unanswered_in", _sum_w(c[(c.direction == "in") & ~answered]))
    short = c_out[c_out.duration_s < 30]
    put("call_max_short_out_single_number",
        short.w.groupby([short.case_id, short.counterparty]).sum().groupby(level=0).max())
    put("call_out_night_unanswered", _sum_w(c[(c.direction == "out") & ~answered & (c_hour < NIGHT_HOURS)]))

    # ------------------------------------------------------------------ web
    w = tables["web"]
    put("web_total_visits", _sum_w(w))
    put("web_distinct_domains", _p_any(w, ["case_id", "domain"]).groupby(level=0).sum())
    put("web_social_share", (w.w * w.domain.isin(SOCIAL_DOMAINS)).groupby(w.case_id).sum() / _sum_w(w))
    put("web_hacktool_visits", _sum_w(w[w.domain.isin(HACKTOOL_BLOCKLIST)]))

    # --------------------------------------------------------------- logins
    lg = tables["login"]
    ok = lg.success.astype(str).str.lower() == "true"
    put("login_total", _sum_w(lg))
    put("login_failed_total", _sum_w(lg[~ok]))
    put("login_failed_distinct_accounts", _p_any(lg[~ok], ["case_id", "account"]).groupby(level=0).sum())
    put("login_distinct_ip_countries", _p_any(lg, ["case_id", "ip_country"]).groupby(level=0).sum())

    # --------------------------------------------------------- transactions
    t = tables["txn"]
    t_out, t_in = t[t.direction == "out"], t[t.direction == "in"]
    put("txn_out_count", _sum_w(t_out))
    put("txn_out_amount_log", np.log1p((t_out.w * t_out.amount).groupby(t_out.case_id).sum()))
    put("txn_in_count", _sum_w(t_in))
    put("txn_in_new_counterparty", _new_counterparty(t))

    # ------------------------------------------------- cross-artifact feature
    parts = [pd.DataFrame({"case_id": d.case_id, "w": d.w, "night": _ts(d).dt.hour < NIGHT_HOURS})
             for d in tables.values() if len(d)]
    h = pd.concat(parts)
    put("night_activity_share", (h.w * h.night).groupby(h.case_id).sum() / h.w.groupby(h.case_id).sum())
    return out[FEATURES]


def _new_counterparty(t: pd.DataFrame) -> pd.Series:
    """Expected number of incoming payments whose counterparty appears in NO other
    transaction of the case (N1: n_seen == 1)."""
    if not len(t):
        return pd.Series(dtype=float)
    g = [t.case_id, t.counterparty]
    is_one = t.w >= 1.0
    n_one = is_one.groupby(g).transform("sum")                           # records with w = 1
    logq = np.log((1.0 - t.w).where(~is_one, 1.0).clip(lower=1e-300))    # log(1-w), 0 for w = 1
    sum_logq = logq.groupby(g).transform("sum")
    others_one = n_one - is_one.astype(int)
    others = np.where(others_one > 0, 0.0, np.exp(sum_logq - logq))
    val = pd.Series(t.w.to_numpy() * others, index=t.index)
    inc = t.direction == "in"
    return val[inc].groupby(t.case_id[inc]).sum()
