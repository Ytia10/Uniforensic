"""Write one N1 case into real SQLite app databases, delete artifacts the way a
user would, and map carved records back to unified-schema artifacts.

One database file per artifact table, like the separate app databases on a phone
(messages, call log, browser history, a login log, a payments app):

  messages        - the teammate's N3 'sms' schema, unchanged
  calls           - the teammate's N3 'calls' schema, unchanged
  browser_history - the teammate's N3 'browser' schema, unchanged
  logins          - new, same style (N1 has login artifacts, N3 had no schema for them)
  transactions    - new, same style (amount stored in paise as INTEGER, the usual
                    way payment apps store money; REAL is not in the N3 feature f3)

The PRAGMAs are the teammate's main-benchmark settings: page_size 4096,
secure_delete OFF (SQLite's default), auto_vacuum NONE, journal_mode DELETE.

Inserts and deletions happen on ONE timeline in time order (not "insert all, then
delete"), so rows deleted early can be partly or wholly overwritten by later
inserts, exactly as on a phone that stays in use after the clean-up.
"""
from __future__ import annotations

import calendar
import os
import re
import sqlite3
from datetime import datetime, timezone

import pandas as pd

OWNER = "+910000000000"          # the seized phone's own number (fixed, fictional)

SCHEMAS = {
    "sms": {"table_name": "messages",
            "columns": ["id", "sender", "recipient", "body", "timestamp", "status"],
            "column_types": ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER", "INTEGER"],
            "sql": "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender TEXT NOT NULL, "
                   "recipient TEXT NOT NULL, body TEXT NOT NULL, timestamp INTEGER NOT NULL, "
                   "status INTEGER NOT NULL)"},
    "call": {"table_name": "calls",
             "columns": ["id", "phone_number", "duration_sec", "timestamp", "call_type"],
             "column_types": ["INTEGER", "TEXT", "INTEGER", "INTEGER", "TEXT"],
             "sql": "CREATE TABLE calls (id INTEGER PRIMARY KEY AUTOINCREMENT, phone_number TEXT NOT NULL, "
                    "duration_sec INTEGER NOT NULL, timestamp INTEGER NOT NULL, call_type TEXT NOT NULL)"},
    "web": {"table_name": "browser_history",
            "columns": ["id", "url", "title", "visit_time", "visit_count"],
            "column_types": ["INTEGER", "TEXT", "TEXT", "INTEGER", "INTEGER"],
            "sql": "CREATE TABLE browser_history (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL, "
                   "title TEXT NOT NULL, visit_time INTEGER NOT NULL, visit_count INTEGER NOT NULL)"},
    "login": {"table_name": "logins",
              "columns": ["id", "service", "account", "ip_country", "timestamp", "success"],
              "column_types": ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER", "INTEGER"],
              "sql": "CREATE TABLE logins (id INTEGER PRIMARY KEY AUTOINCREMENT, service TEXT NOT NULL, "
                     "account TEXT NOT NULL, ip_country TEXT NOT NULL, timestamp INTEGER NOT NULL, "
                     "success INTEGER NOT NULL)"},
    "txn": {"table_name": "transactions",
            "columns": ["id", "direction", "counterparty", "amount_paise", "timestamp"],
            "column_types": ["INTEGER", "TEXT", "TEXT", "INTEGER", "INTEGER"],
            "sql": "CREATE TABLE transactions (id INTEGER PRIMARY KEY AUTOINCREMENT, direction TEXT NOT NULL, "
                   "counterparty TEXT NOT NULL, amount_paise INTEGER NOT NULL, timestamp INTEGER NOT NULL)"},
}
TABLES = list(SCHEMAS)            # sms, call, web, login, txn (N1's ARTIFACT_TYPES order)
TS_MIN, TS_MAX = 1262304000, 2051222400   # 2010..2035, the range the N3 feature f6 accepts
_URL_RE = re.compile(r"^https://([a-z0-9.-]+)/$")


def schema_info(table: str) -> dict:
    """The dict the teammate's SQLiteCarver expects."""
    s = SCHEMAS[table]
    return {k: s[k] for k in ("table_name", "columns", "column_types")}


# ---------------------------------------------------------------- time helpers
def to_epoch(ts: str) -> int:
    return calendar.timegm(datetime.fromisoformat(ts).timetuple())


def from_epoch(v: int) -> str:
    return datetime.fromtimestamp(v, tz=timezone.utc).replace(tzinfo=None).isoformat()


def _title(domain: str) -> str:
    return domain.split(".")[0].replace("-", " ").title()


# ------------------------------------------------- artifact <-> database row
def to_row(table: str, a) -> tuple:
    """Unified-schema artifact (a namedtuple / Series) -> stored column values (no id)."""
    if table == "sms":
        out = a.direction == "out"
        return (OWNER if out else str(a.counterparty), str(a.counterparty) if out else OWNER,
                str(a.text), to_epoch(a.timestamp), 2 if out else 1)
    if table == "call":
        answered = str(a.answered).lower() == "true"
        kind = "OUTGOING" if a.direction == "out" else ("INCOMING" if answered else "MISSED")
        return (str(a.counterparty), int(a.duration_s), to_epoch(a.timestamp), kind)
    if table == "web":
        return (f"https://{a.domain}/", _title(str(a.domain)), to_epoch(a.timestamp), 1)
    if table == "login":
        return (str(a.service), str(a.account), str(a.ip_country), to_epoch(a.timestamp),
                1 if str(a.success).lower() == "true" else 0)
    if table == "txn":
        return (str(a.direction), str(a.counterparty), int(round(float(a.amount) * 100)),
                to_epoch(a.timestamp))
    raise ValueError(table)


def _ok_ts(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and TS_MIN <= v <= TS_MAX


def from_row(table: str, v: tuple) -> dict | None:
    """Carved record values -> unified-schema artifact fields, or None if the record
    cannot be a valid row of this table (wrong types, inconsistent fields).
    This is what an examiner's tool does before using a carved record."""
    if not all(isinstance(x, (int, str)) and not isinstance(x, bool) for x in v):
        return None
    if table == "sms":
        snd, rcv, body, ts, st = v
        if not (isinstance(snd, str) and isinstance(rcv, str) and isinstance(body, str) and _ok_ts(ts)):
            return None
        if st == 2 and snd == OWNER and rcv != OWNER:
            return dict(timestamp=from_epoch(ts), direction="out", counterparty=rcv, text=body)
        if st == 1 and rcv == OWNER and snd != OWNER:
            return dict(timestamp=from_epoch(ts), direction="in", counterparty=snd, text=body)
        return None
    if table == "call":
        num, dur, ts, kind = v
        if not (isinstance(num, str) and isinstance(dur, int) and dur >= 0 and _ok_ts(ts)):
            return None
        if kind == "OUTGOING":
            return dict(timestamp=from_epoch(ts), direction="out", counterparty=num,
                        answered=dur > 0, duration_s=dur)
        if kind == "INCOMING" and dur > 0:
            return dict(timestamp=from_epoch(ts), direction="in", counterparty=num, answered=True,
                        duration_s=dur)
        if kind == "MISSED" and dur == 0:
            return dict(timestamp=from_epoch(ts), direction="in", counterparty=num, answered=False,
                        duration_s=0)
        return None
    if table == "web":
        url, title, ts, cnt = v
        m = _URL_RE.match(url) if isinstance(url, str) else None
        if not (m and isinstance(title, str) and _ok_ts(ts) and isinstance(cnt, int) and cnt >= 1):
            return None
        return dict(timestamp=from_epoch(ts), domain=m.group(1))
    if table == "login":
        svc, acct, ctry, ts, ok = v
        if not (isinstance(svc, str) and isinstance(acct, str) and isinstance(ctry, str)
                and _ok_ts(ts) and ok in (0, 1)):
            return None
        return dict(timestamp=from_epoch(ts), service=svc, account=acct, ip_country=ctry,
                    success=bool(ok))
    if table == "txn":
        d, cp, paise, ts = v
        if not (d in ("in", "out") and isinstance(cp, str) and isinstance(paise, int)
                and paise >= 0 and _ok_ts(ts)):
            return None
        return dict(timestamp=from_epoch(ts), direction=d, counterparty=cp, amount=paise / 100)
    raise ValueError(table)


def normalise(table: str, values: list) -> tuple | None:
    """Carver output -> stored values without the rowid. With INTEGER PRIMARY KEY the
    record stores NULL for 'id'; the teammate's carver may return it or skip it."""
    n = len(SCHEMAS[table]["columns"]) - 1
    if len(values) == n + 1 and values[0] is None:
        values = values[1:]
    return tuple(values) if len(values) == n else None


# ------------------------------------------------------------- build one DB
def build_db(path: str, table: str, events: list[tuple]) -> None:
    """events: time-ordered list of ('ins', key, row) and ('del', key, None).
    Executes them in order on a fresh database with the teammate's PRAGMAs."""
    for f in (path, path + "-journal", path + "-wal", path + "-shm"):
        if os.path.exists(f):
            os.remove(f)
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("PRAGMA page_size = 4096")
    con.execute("PRAGMA secure_delete = 0")
    con.execute("PRAGMA auto_vacuum = 0")
    con.execute("PRAGMA journal_mode = DELETE")
    con.execute("PRAGMA synchronous = OFF")          # speed only; does not change page layout
    con.execute(SCHEMAS[table]["sql"])
    cols = SCHEMAS[table]["columns"][1:]
    ins = (f"INSERT INTO {SCHEMAS[table]['table_name']} ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' * len(cols))})")
    dele = f"DELETE FROM {SCHEMAS[table]['table_name']} WHERE id = ?"
    rowid = {}
    con.execute("BEGIN")
    for kind, key, row in events:
        if kind == "ins":
            rowid[key] = con.execute(ins, row).lastrowid
        else:
            con.execute(dele, (rowid[key],))
    con.execute("COMMIT")
    con.close()


def read_live_rows(path: str, table: str) -> set[tuple]:
    """Live rows, read through a READ-ONLY, immutable connection (the evidence file is
    never opened for writing, and no journal or WAL is created or replayed)."""
    uri = f"file:{os.path.abspath(path)}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    cols = ", ".join(SCHEMAS[table]["columns"][1:])
    rows = set(con.execute(f"SELECT {cols} FROM {SCHEMAS[table]['table_name']}").fetchall())
    con.close()
    return rows


def artifacts_frame(table: str, recs: list[dict], case_id: str) -> pd.DataFrame:
    return pd.DataFrame([{"case_id": case_id, **r} for r in recs])
