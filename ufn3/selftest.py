"""Self-tests run by n3_00_check_env.py. Each builds small databases with the real
SQLite engine and checks the reader / carver against what SQLite itself reports."""
from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import apsw

from .carver import carve, group_candidates
from .common import row_key
from .sqlite_format import encode_varint, open_image, read_varint


def _connect(path: Path, page_size: int, journal: str, secure: str, vacuum: str):
    con = apsw.Connection(str(path))
    con.config(apsw.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1)
    cur = con.cursor()
    for q in [f"PRAGMA page_size={page_size}", f"PRAGMA auto_vacuum={vacuum}",
              f"PRAGMA journal_mode={journal}", f"PRAGMA secure_delete={secure}"]:
        list(cur.execute(q))
    return con, cur


def _read(path: Path) -> tuple[bytes, bytes | None]:
    w = Path(f"{path}-wal")
    return path.read_bytes(), (w.read_bytes() if w.exists() else None)


def test_varints() -> None:
    for v in [0, 1, 127, 128, 240, 2287, 16383, 16384, 2 ** 32, 2 ** 56 - 1, 2 ** 56, 2 ** 63, 2 ** 64 - 1]:
        b = encode_varint(v)
        assert read_varint(b, 0, len(b)) == (v, len(b)), v


def test_reader_matches_sqlite(n_db: int = 12) -> int:
    """Live rows read from raw bytes == rows returned by SQLite, across page sizes,
    WAL / rollback journals, auto_vacuum, overflow pages, UTF-8, BLOBs and floats."""
    tmp = Path(tempfile.mkdtemp(prefix="ufn3_"))
    try:
        for seed in range(n_db):
            r = random.Random(seed)
            ps = [512, 1024, 4096, 8192, 65536][seed % 5]
            journal = ["DELETE", "WAL"][seed % 2]
            path = tmp / f"t{seed}.db"
            con, cur = _connect(path, ps, journal, ["OFF", "FAST", "ON"][seed % 3],
                                ["NONE", "FULL", "INCREMENTAL"][(seed // 3) % 3])
            cur.execute("CREATE TABLE sms(_id INTEGER PRIMARY KEY, address TEXT, date INTEGER, body TEXT, f REAL, b BLOB)")
            cur.execute("CREATE TABLE t3('id' INT NOT NULL, x TEXT, y INTEGER)")
            cur.execute("CREATE INDEX i1 ON sms(address)")
            with con:
                for k in range(r.randint(60, 300)):
                    body = "x" * r.choice([1, 10, 100, 1000, 5000]) + str(k)
                    cur.execute("INSERT INTO sms(address,date,body,f,b) VALUES(?,?,?,?,?)",
                                (f"+91{r.randint(10 ** 9, 10 ** 10)}", r.randint(0, 2 ** 45), body,
                                 r.random() if r.random() < .5 else None,
                                 bytes(r.randrange(256) for _ in range(r.randint(0, 40))) if r.random() < .3 else None))
                    cur.execute("INSERT INTO t3 VALUES(?,?,?)", (k, "é漢" * r.randint(0, 5), r.randint(-2 ** 60, 2 ** 60)))
            with con:
                cur.execute("DELETE FROM sms WHERE _id % 3 = 0")
                cur.execute("UPDATE t3 SET x='upd' WHERE id % 7 = 0")
            truth = {"sms": sorted(map(tuple, cur.execute("SELECT * FROM sms"))),
                     "t3": sorted(map(tuple, cur.execute("SELECT rowid, * FROM t3")))}
            con.close()
            img = open_image(*_read(path))
            got = {"sms": [], "t3": []}
            for row in img.live:
                got[row.table].append(tuple(row.values) if row.table == "sms" else (row.rowid, *row.values))
            for t in truth:
                assert sorted(got[t]) == truth[t], f"reader differs from SQLite (db {seed}, table {t})"
        return n_db
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_carver_finds_deleted() -> dict:
    """Delete rows under secure_delete=OFF and check every deleted row is recovered,
    via the freeblock hypothesis (first 4 bytes overwritten) among others, and that no
    live row is ever offered as deleted. Under secure_delete=ON nothing may be found."""
    tmp = Path(tempfile.mkdtemp(prefix="ufn3_"))
    out = {}
    try:
        for secure in ("OFF", "ON"):
            path = tmp / f"c_{secure}.db"
            con, cur = _connect(path, 4096, "DELETE", secure, "NONE")
            cur.execute("CREATE TABLE sms(_id INTEGER PRIMARY KEY, address TEXT, date INTEGER, body TEXT)")
            rows = [(f"+9190000{k:05d}", 1_780_000_000_000 + k * 7919, f"message number {k} sent") for k in range(60)]
            with con:
                cur.executemany("INSERT INTO sms(address,date,body) VALUES(?,?,?)", rows)
            gone = [5, 17, 29, 41, 53]
            with con:
                for g in gone:
                    cur.execute("DELETE FROM sms WHERE _id=?", (g,))
            con.close()
            img = open_image(*_read(path))
            recs = group_candidates(img, carve(img)[0])
            want = {row_key([None, *rows[g - 1]], 0) for g in gone}
            live = {row_key([None, *rows[k]], 0) for k in range(60) if k + 1 not in gone}
            found = {r.key for r in recs if not r.live_duplicate}
            assert not (found & live), "a live row was offered as deleted"
            hyps = {r.best.hypothesis for r in recs if r.key in want}
            out[secure] = (len(found & want), hyps)
        assert out["OFF"][0] == 5 and "freeblock" in out["OFF"][1], out
        assert out["ON"][0] == 0, out
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_journal_recovery() -> int:
    """PERSIST rollback journal + secure_delete=ON: the deleted rows are wiped from the
    database file but their old pages are still in the journal - all must be found."""
    tmp = Path(tempfile.mkdtemp(prefix="ufn3_"))
    try:
        path = tmp / "j.db"
        con, cur = _connect(path, 4096, "PERSIST", "ON", "NONE")
        cur.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT, b INTEGER)")
        with con:
            cur.executemany("INSERT INTO t(a,b) VALUES(?,?)", [(f"row text {i}", i * 1000) for i in range(300)])
        with con:
            cur.execute("DELETE FROM t WHERE id IN (5, 77, 150, 299)")
        con.close()
        jrn = Path(f"{path}-journal").read_bytes()
        img = open_image(path.read_bytes(), None, jrn)
        cands, info = carve(img)
        recs = group_candidates(img, cands, info["bounds"])
        want = {row_key([None, f"row text {i - 1}", (i - 1) * 1000], 0) for i in (5, 77, 150, 299)}
        got = {r.key for r in recs if not r.live_duplicate and r.best.source == "journal"}
        assert want <= got, "deleted rows not recovered from the rollback journal"
        return len(want & got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_all() -> list[str]:
    test_varints()
    n = test_reader_matches_sqlite()
    c = test_carver_finds_deleted()
    j = test_journal_recovery()
    return ["varint encode/decode round-trips",
            f"live rows read from raw bytes == SQLite's own answer ({n} databases, 5 page sizes, WAL + rollback)",
            f"carver recovers 5/5 deleted rows with secure_delete=OFF (hypotheses: {sorted(c['OFF'][1])}), "
            f"0/5 with secure_delete=ON, and never offers a live row as deleted",
            f"rollback journal (PERSIST, secure_delete=ON): {j}/4 deleted rows recovered from old page images"]


__all__ = ["run_all"]
