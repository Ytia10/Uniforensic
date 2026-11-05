"""Build realistic SQLite databases from N1 cases, delete rows, and log the truth.

Every N1 sms / call / web artifact becomes rows of app-like tables. The case's
owner then deletes artifacts over time: planted evidence with high probability (a
suspect cleaning the phone), other artifacts at a low per-case rate. Inserts and
deletes are replayed in time order and committed in small batches, so later
activity overwrites part of the freed space - as on a real device.

Ground truth = every row version ever written, with its N1 artifact_id and whether
it is still live at the end. Nothing about the truth is stored in the database files.

The SQLite engine is APSW, a pinned package that ships its OWN copy of SQLite (version
printed by n3_00_check_env.py), so every computer builds byte-identical files
whatever SQLite the operating system has.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

import apsw
import numpy as np

from ufn1.generator import HACK_DOMAINS, TECH_DOMAINS

from .sqlite_schema import parse_create_table

# App schemas. Android and Chromium: column names, types and constraints copied from
# the real apps (Android tables keep a subset of their columns; Chromium tables follow
# Chromium's history.38 test schema exactly, components/test/data/history/history.38.sql).
# "pay": a payment-app database whose schema is OURS (no public standard exists). It is
# included on purpose: its tables use the common "id INT" pattern, which is NOT a rowid
# alias, so a damaged record's first column cannot be inferred from the schema - the
# hard case that also appears in the third-party corpus of step 7.
DATABASES = {
    "sms": {"file": "mmssms.db", "ddl": [
        "CREATE TABLE sms (_id INTEGER PRIMARY KEY, thread_id INTEGER, address TEXT, person INTEGER, "
        "date INTEGER, date_sent INTEGER DEFAULT 0, protocol INTEGER, read INTEGER DEFAULT 0, "
        "status INTEGER DEFAULT -1, type INTEGER, reply_path_present INTEGER, subject TEXT, body TEXT, "
        "service_center TEXT, locked INTEGER DEFAULT 0, error_code INTEGER DEFAULT 0, seen INTEGER DEFAULT 0)",
        "CREATE INDEX typeThreadIdIndex ON sms (type, thread_id)"]},
    "call": {"file": "calllog.db", "ddl": [
        "CREATE TABLE calls (_id INTEGER PRIMARY KEY AUTOINCREMENT, number TEXT, date INTEGER, "
        "duration INTEGER, type INTEGER, new INTEGER, name TEXT, numbertype INTEGER, numberlabel TEXT, "
        "countryiso TEXT, geocoded_location TEXT)"]},
    "web": {"file": "History", "ddl": [
        "CREATE TABLE urls(id INTEGER PRIMARY KEY AUTOINCREMENT,url LONGVARCHAR,title LONGVARCHAR,"
        "visit_count INTEGER DEFAULT 0 NOT NULL,typed_count INTEGER DEFAULT 0 NOT NULL,"
        "last_visit_time INTEGER NOT NULL,hidden INTEGER DEFAULT 0 NOT NULL)",
        "CREATE TABLE visits(id INTEGER PRIMARY KEY,url INTEGER NOT NULL,visit_time INTEGER NOT NULL,"
        "from_visit INTEGER,transition INTEGER DEFAULT 0 NOT NULL,segment_id INTEGER,"
        "visit_duration INTEGER DEFAULT 0 NOT NULL)",
        "CREATE TABLE keyword_search_terms (keyword_id INTEGER NOT NULL,url_id INTEGER NOT NULL,"
        "lower_term LONGVARCHAR NOT NULL,term LONGVARCHAR NOT NULL)",
        "CREATE INDEX urls_url_index ON urls (url)",
        "CREATE INDEX visits_url_index ON visits (url)",
        "CREATE INDEX keyword_search_terms_index1 ON keyword_search_terms (keyword_id, lower_term)"]},
    "pay": {"file": "payments.db", "ddl": [
        "CREATE TABLE transactions (txn_id INT NOT NULL, created_at INTEGER NOT NULL, "
        "direction INTEGER NOT NULL, counterparty TEXT NOT NULL, amount REAL NOT NULL, "
        "status INTEGER NOT NULL DEFAULT 1)",
        "CREATE TABLE ledger (txn_id INT NOT NULL, balance_paise INTEGER NOT NULL, updated_at INTEGER NOT NULL)",
        "CREATE INDEX transactions_time ON transactions (created_at)"]},
}
_EPOCH = datetime(1970, 1, 1)
_WEBKIT_OFFSET_US = 11644473600 * 1_000_000          # 1601-01-01 -> 1970-01-01, in microseconds
_PATHS = ["", "login", "watch?v=", "post/", "item/", "news/", "profile/", "download/"]
_TERMS = {"benign": ["Weather today", "cheap flights Goa", "paneer recipe", "cricket score", "bank holidays",
                     "movie tickets", "train status", "phone offers"],
          "tech": ["python sort list", "git rebase", "docker compose port", "regex email"],
          "hack": ["stresser panel", "exploit download", "combo list", "rat builder", "free cvv"]}
_TRANSITIONS = [805306368, 805306369, 838860801]      # Chromium page-transition values (link / typed)


def tables_of(db: str) -> dict:
    """name -> TableSchema of the user tables of a database type (parsed from its DDL)."""
    out = {}
    for q in DATABASES[db]["ddl"]:
        if q.startswith("CREATE TABLE"):
            name = q.split("(")[0].split()[2]
            out[name] = parse_create_table(name, q)
    return out


def _ms(ts: str) -> int:
    return int((datetime.fromisoformat(ts) - _EPOCH).total_seconds() * 1000)


def artifact_rows(db: str, a: dict, rng: np.random.Generator, ctx: dict) -> list[tuple[str, dict]]:
    """Rows to insert for one N1 artifact: [(table, {column: value})], parents first.
    A value of ('ref', table) means 'the rowid just inserted into that table'."""
    t = _ms(a["timestamp"])
    if db == "sms":
        inbox = a["direction"] == "in"
        thread = ctx.setdefault("threads", {}).setdefault(a["counterparty"], len(ctx["threads"]) + 1)
        return [("sms", {"thread_id": thread, "address": a["counterparty"], "person": None, "date": t,
                         "date_sent": t - int(rng.integers(500, 9000)) if inbox else 0,
                         "protocol": 0 if inbox else None, "read": 1, "status": -1, "type": 1 if inbox else 2,
                         "reply_path_present": 0 if inbox else None, "subject": None, "body": a["text"],
                         "service_center": ("+9198" + "".join(map(str, rng.integers(0, 10, 8)))) if inbox else None,
                         "locked": 0, "error_code": 0, "seen": 1})]
    if db == "call":
        incoming, answered = a["direction"] == "in", str(a["answered"]) == "True"
        return [("calls", {"number": a["counterparty"], "date": t,
                           "duration": int(float(a["duration_s"])) if answered else 0,
                           "type": (1 if answered else 3) if incoming else 2,
                           "new": 0 if answered or not incoming else 1, "name": None,
                           "numbertype": int(rng.choice([0, 1, 2])), "numberlabel": None,
                           "countryiso": "IN", "geocoded_location": "India"})]
    if db == "web":
        dom = a["domain"]
        group = "hack" if dom in HACK_DOMAINS else ("tech" if dom in TECH_DOMAINS else "benign")
        term = str(rng.choice(_TERMS[group])) if rng.random() < 0.3 else None
        path = (f"search?q={term.replace(' ', '+')}" if term else
                str(rng.choice(_PATHS)) + "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz0123456789"),
                                                             size=int(rng.integers(0, 12)))))
        wk = t * 1000 + _WEBKIT_OFFSET_US
        visits = int(rng.integers(1, 6))
        rows = [("urls", {"url": f"https://{dom}/{path}", "title": dom.split(".")[0].replace("-", " ").title(),
                          "visit_count": visits, "typed_count": int(rng.integers(0, visits + 1)),
                          "last_visit_time": wk, "hidden": 0}),
                ("visits", {"url": ("ref", "urls"), "visit_time": wk, "from_visit": 0,
                            "transition": int(rng.choice(_TRANSITIONS)), "segment_id": 0,
                            "visit_duration": int(rng.integers(0, 600_000_000))})]
        if term:
            rows.append(("keyword_search_terms", {"keyword_id": 2, "url_id": ("ref", "urls"),
                                                  "lower_term": term.lower(), "term": term}))
        return rows
    if db == "pay":
        txn_id = ctx.setdefault("txn_id", int(rng.integers(100_000, 900_000))) + 1
        ctx["txn_id"] = txn_id
        amount = round(float(a["amount"]), 2)
        sign = 1 if a["direction"] == "in" else -1
        ctx["balance"] = ctx.get("balance", int(rng.integers(1_000_00, 500_000_00))) + sign * round(amount * 100)
        return [("transactions", {"txn_id": txn_id, "created_at": t, "direction": 1 if sign > 0 else 2,
                                  "counterparty": a["counterparty"], "amount": amount,
                                  "status": int(rng.choice([1, 1, 1, 2]))}),
                ("ledger", {"txn_id": txn_id, "balance_paise": ctx["balance"], "updated_at": t + int(rng.integers(1, 5000))})]
    raise ValueError(db)


class _SeededVFS(apsw.VFS):
    """This computer's normal file system, except that SQLite's random numbers (used
    for the WAL 'salt' values) come from a seed. Without it two runs would write
    different salt bytes into every WAL file; with it every database is
    byte-identical on every run and every computer."""

    def __init__(self):
        self.seed = b"\0" * 32
        super().__init__("ufn3_seeded", base="", makedefault=True)

    def xRandomness(self, numbytes: int) -> bytes:
        out, i = b"", 0
        while len(out) < numbytes:
            out += hashlib.sha256(self.seed + i.to_bytes(4, "big")).digest()
            i += 1
        return out[:numbytes]


_VFS: list = []


def _seed_sqlite(seed: int) -> None:
    """sqlite3_randomness(0, ...) re-seeds SQLite's generator from the default VFS."""
    if not _VFS:
        _VFS.append(_SeededVFS())
    _VFS[0].seed = hashlib.sha256(str(seed).encode()).digest()
    apsw.randomness(0)


def _pragmas(cur, cond: dict) -> None:
    for q in [f"PRAGMA page_size={cond['page_size']}", f"PRAGMA auto_vacuum={cond['auto_vacuum']}",
              f"PRAGMA journal_mode={cond['journal']}", f"PRAGMA secure_delete={cond['secure_delete']}",
              "PRAGMA synchronous=OFF"]:             # no fsync: faster, same bytes (checked in step 2)
        list(cur.execute(q))


def build_database(db: str, artifacts: list[dict], case: dict, cond_name: str, cond: dict,
                   cfg3: dict, rng: np.random.Generator, out_dir: Path) -> dict:
    """Create one database in out_dir and return its ground truth."""
    spec, schemas = DATABASES[db], tables_of(db)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / spec["file"]
    for suffix in ("", "-wal", "-shm", "-journal"):
        if Path(f"{path}{suffix}").exists():
            os.remove(f"{path}{suffix}")
    dcfg = cfg3["deletion"]
    other_rate = float(rng.uniform(*dcfg["other_delete_prob"]))
    arts = sorted(artifacts, key=lambda a: (a["timestamp"], a["artifact_id"]))
    events = []                                                   # (time_ms, order, kind, index)
    for i, a in enumerate(arts):
        t = _ms(a["timestamp"])
        events.append((t, 0, "insert", i))
        if rng.random() < (dcfg["evidence_delete_prob"] if a["role"] == "evidence" else other_rate):
            events.append((t + int(rng.uniform(*dcfg["delay_hours"]) * 3_600_000), 1, "delete", i))
    events.sort()
    _seed_sqlite(int(rng.integers(0, 2 ** 62)))
    con = apsw.Connection(str(path))
    con.config(apsw.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1)          # keep the WAL, like a live acquisition
    cur = con.cursor()
    _pragmas(cur, cond)
    for q in spec["ddl"]:
        cur.execute(q)
    wal = cond["journal"].upper() == "WAL"
    ckpt_at = int(len(events) * rng.uniform(*cfg3["wal_checkpoint_at"])) if wal else -1
    ctx: dict = {}
    truth = {n: {"columns": s.columns, "ipk": s.ipk, "versions": []} for n, s in schemas.items()}
    made: dict[int, list[tuple[str, int, int]]] = {}              # artifact -> [(table, rowid, version index)]
    pos, lo, hi = 0, dcfg["transaction_size"][0], dcfg["transaction_size"][1]
    while pos < len(events):
        batch = events[pos:pos + int(rng.integers(lo, hi + 1))]
        with con:
            for _t, _o, kind, i in batch:
                a = arts[i]
                if kind == "insert":
                    last: dict[str, int] = {}
                    for table, row in artifact_rows(db, a, rng, ctx):
                        row = {k: (last[v[1]] if isinstance(v, tuple) else v) for k, v in row.items()}
                        cols = list(row)
                        cur.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                    [row[c] for c in cols])
                        rid = con.last_insert_rowid()
                        last[table] = rid
                        sch = schemas[table]
                        vals = [rid if j == sch.ipk else row.get(c) for j, c in enumerate(sch.columns)]
                        made.setdefault(i, []).append((table, rid, len(truth[table]["versions"])))
                        truth[table]["versions"].append({"rowid": rid, "values": vals, "artifact_id": a["artifact_id"],
                                                         "role": a["role"], "pattern_id": a["pattern_id"],
                                                         "deleted": False})
                else:
                    for table, rid, k in made[i]:
                        cur.execute(f"DELETE FROM {table} WHERE rowid = ?", (rid,))
                        truth[table]["versions"][k]["deleted"] = True
        if pos <= ckpt_at < pos + len(batch):
            list(cur.execute("PRAGMA wal_checkpoint(PASSIVE)"))
        pos += len(batch)
    for table, t in truth.items():                                # bookkeeping check against SQLite
        live = {r[0] for r in cur.execute(f"SELECT rowid FROM {table}")}
        assert live == {v["rowid"] for v in t["versions"] if not v["deleted"]}, f"truth mismatch in {table}"
    con.close()
    if Path(f"{path}-shm").exists():
        os.remove(f"{path}-shm")                                  # shared-memory index, not evidence
    return {"case_id": case["case_id"], "label": case["label"], "tier": case["tier"], "db": db,
            "condition": cond_name, "file": spec["file"], "tables": truth}
