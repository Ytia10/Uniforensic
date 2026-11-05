"""Read a table's columns from its CREATE TABLE statement (as stored in sqlite_master).

What the carver needs per table: column names, each column's type AFFINITY (the
rule SQLite itself uses to decide what a column prefers to store), and whether
one column is an alias for the rowid ("INTEGER PRIMARY KEY"). An alias column
is stored as NULL inside the record; its real value is the cell's rowid.

Affinity rules, in order (SQLite docs, "Datatypes In SQLite", section 3.1):
  1. declared type contains "INT"                -> INTEGER
  2. contains "CHAR", "CLOB" or "TEXT"           -> TEXT
  3. contains "BLOB", or no type at all          -> BLOB
  4. contains "REAL", "FLOA" or "DOUB"           -> REAL
  5. anything else                               -> NUMERIC
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CONSTRAINT_WORDS = {"CONSTRAINT", "PRIMARY", "NOT", "NULL", "UNIQUE", "CHECK", "DEFAULT",
                     "COLLATE", "REFERENCES", "GENERATED", "AS"}
_TABLE_CONSTRAINT_START = {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"}


def affinity(declared_type: str) -> str:
    t = declared_type.upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or t.strip() == "":
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


@dataclass
class TableSchema:
    name: str
    columns: list[str]
    types: list[str]                    # declared types, as written
    affinities: list[str]
    ipk: int | None                     # index of the INTEGER PRIMARY KEY alias column, or None
    without_rowid: bool = False
    root_page: int = 0
    sql: str = ""
    stored: list[int] = field(default_factory=list)   # indexes of columns stored in the record

    @property
    def n_cols(self) -> int:
        return len(self.stored)


def _split_top_level(s: str) -> list[str]:
    """Split on commas that are not inside parentheses or quotes."""
    parts, depth, cur, quote = [], 0, [], None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'`[":
            quote = "]" if ch == "[" else ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)


def _read_name(s: str) -> tuple[str, str]:
    """Read one identifier from the start of s (quoted or bare). Returns (name, rest)."""
    s = s.lstrip()
    if not s:
        return "", ""
    q = s[0]
    if q in "\"'`[":
        close = "]" if q == "[" else q
        i, out = 1, []
        while i < len(s):
            if s[i] == close:
                if close != "]" and i + 1 < len(s) and s[i + 1] == close:   # doubled quote
                    out.append(close)
                    i += 2
                    continue
                return "".join(out), s[i + 1:]
            out.append(s[i])
            i += 1
        return "".join(out), ""
    m = re.match(r"[^\s(,]+", s)
    return m.group(0), s[m.end():]


def _tokens(s: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", s)


def parse_create_table(name: str, sql: str, root_page: int = 0) -> TableSchema | None:
    """Columns of a CREATE TABLE statement, or None if it cannot be read
    (virtual tables, CREATE TABLE ... AS SELECT)."""
    if not sql:
        return None
    body_sql = _strip_comments(sql)
    lo, hi = body_sql.find("("), body_sql.rfind(")")
    if lo < 0 or hi < lo or re.match(r"\s*CREATE\s+VIRTUAL", body_sql, re.I):
        return None
    tail = body_sql[hi + 1:].upper()
    without_rowid = "WITHOUT" in tail and "ROWID" in tail
    cols, types, pk_cols, pk_desc, generated = [], [], [], {}, set()
    for part in _split_top_level(body_sql[lo + 1:hi]):
        first = _tokens(part)[:1]
        if first and first[0].upper() in _TABLE_CONSTRAINT_START and not part.lstrip()[0] in "\"'`[":
            up = part.upper()
            m = re.search(r"PRIMARY\s+KEY\s*\((.*)\)", part, re.I | re.S)
            if m and "PRIMARY" in up:
                pk_cols = [_read_name(c)[0] for c in _split_top_level(m.group(1))]
            continue
        cname, rest = _read_name(part)
        toks = _tokens(rest)
        tparts, i = [], 0
        while i < len(toks) and toks[i].upper() not in _CONSTRAINT_WORDS:
            tparts.append(toks[i])
            i += 1
        ctype = " ".join(tparts).replace("( ", "(").replace(" )", ")")
        cons = [t.upper() for t in toks[i:]]
        if "PRIMARY" in cons:
            pk_cols = [cname]
            k = cons.index("PRIMARY")
            pk_desc[cname] = len(cons) > k + 2 and cons[k + 2] == "DESC"
        if "GENERATED" in cons or ("AS" in cons and "VIRTUAL" in cons):
            if "STORED" not in cons:
                generated.add(len(cols))
        cols.append(cname)
        types.append(ctype)
    if not cols:
        return None
    ipk = None
    if not without_rowid and len(pk_cols) == 1 and pk_cols[0] in cols:
        j = cols.index(pk_cols[0])
        if types[j].strip().upper() == "INTEGER" and not pk_desc.get(pk_cols[0], False):
            ipk = j
    stored = [j for j in range(len(cols)) if j not in generated]
    return TableSchema(name=name, columns=cols, types=types,
                       affinities=[affinity(t) for t in types], ipk=ipk,
                       without_rowid=without_rowid, root_page=root_page, sql=sql, stored=stored)
