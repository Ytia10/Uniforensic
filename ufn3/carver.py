"""Structural carver: find every place where a deleted record could be, decode it.

Where deleted data can survive (SQLite file-format spec + PRAGMA docs):
  in a live table page   freeblocks (a freed cell, first 4 bytes overwritten)
                         unallocated area (freed cells next to the content area)
  in older page copies   freelist pages, pages past the end of the database,
                         pages superseded by a newer WAL frame, old / uncommitted
                         WAL frames, page images in a rollback journal. These
                         copies still hold whole cells.
Five hypotheses ("ways the bytes could be a record") are tried:
  old_cell         an allocated cell of an OLD copy of a table page (rowid known)
  cell             payload-length + rowid + record, all intact (rowid known)
  record           an intact record whose cell prefix is lost (rowid unknown)
  freeblock        the first 4 bytes were overwritten by a freeblock header: the record
                   header started 2 or 3 bytes into the freeblock; the lost header-length
                   byte is recomputed, and a lost first serial type is taken from the
                   schema (rowid alias = NULL) or from the freeblock size; if the size
                   no longer fits one cell, every number type is tried
  stale_freeblock  the same, inside a merged freeblock, at an old freeblock header
                   left behind when two neighbouring cells were freed
Every hypothesis must decode strictly into exactly the table's column count. The
carver deliberately proposes MORE candidates than are true; the ranker decides.
The candidates of one database are then grouped by decoded content: a group is one
RECOVERED RECORD, and its features summarise all copies that support it.
Nothing here looks at the ground truth.
"""
from __future__ import annotations

import hashlib
import math
import struct
from collections import Counter
from dataclasses import dataclass, field

from .common import norm_value, row_key
from .sqlite_format import (LEAF_TABLE, FormatError, Record, SQLiteImage, cell_pointers, decode_value,
                            freeblocks, full_row, local_payload, page_header, parse_record, read_overflow,
                            read_varint, serial_len)
from .sqlite_schema import TableSchema

SOURCES = ["current", "freelist", "orphan", "superseded_main", "superseded_wal", "wal_uncommitted",
           "wal_stale", "beyond_eof", "journal"]
HYPOTHESES = ["old_cell", "cell", "record", "freeblock", "stale_freeblock"]
REGIONS = ["freeblock", "unallocated", "cell_area", "raw_page"]


@dataclass
class Candidate:
    table: str
    values: list                 # one per declared column (rowid in the alias column if known)
    rowid: int | None
    hypothesis: str
    region: str
    source: str
    pgno: int
    start: int                   # offset in the page where the hypothesis starts
    end: int                     # offset just after the last value byte
    region_end: int
    serial_types: list[int]
    owner_known: bool
    payload_ok: float            # 1 = cell payload length equals the record length, nan = n/a
    size_pinned: float           # freeblock hypothesis: freeblock size fits exactly (1/0), else nan
    overflow: int                # 0 = fits on the page, 1 = overflow chain followed
    raw: bytes = b""             # record bytes (header + body, local part)
    page_id: int = 0             # which carved page copy it came from


@dataclass
class _Ctx:
    img: SQLiteImage
    tables: dict[str, TableSchema]
    cap: int
    out: list = field(default_factory=list)
    n_overflow_missing: int = 0
    page_id: int = 0
    bounds: dict = field(default_factory=dict)     # page_id -> offsets where structures start/end

    def new_page(self, marks) -> None:
        self.page_id += 1
        self.bounds[self.page_id] = set(marks)


# ---------------------------------------------------------------- hypotheses
def _finish_record(ctx: _Ctx, page: bytes, r: int, limit: int, sch: TableSchema) -> tuple[Record, bytes, int] | None:
    """Strict parse of a record at page[r:], following an overflow chain if the record's
    own header says it does not fit on the page. Returns (record, raw bytes, overflow flag)."""
    U = ctx.img.usable
    try:
        hlen, j = read_varint(page, r, limit)
        if hlen < sch.n_cols + 1 or hlen > 9 * sch.n_cols + 1 or r + hlen > limit:
            return None
        sts, k = [], j
        while k < r + hlen:
            st, k = read_varint(page, k, r + hlen)
            if st in (10, 11):
                return None
            sts.append(st)
        if k != r + hlen or len(sts) != sch.n_cols:
            return None
        P = hlen + sum(serial_len(s) for s in sts)
        loc = local_payload(P, U)
        if loc == P:
            if r + P > limit:
                return None
            rec = parse_record(page, r, r + P, ctx.img.encoding, sch.n_cols)
            return rec, bytes(page[r:r + P]), 0
        if r + loc + 4 > limit:
            return None
        first = struct.unpack(">I", page[r + loc:r + loc + 4])[0]
        try:
            rest, _ = read_overflow(ctx.img.pages, first, P - loc, U)
        except FormatError:
            ctx.n_overflow_missing += 1
            return None
        buf = bytes(page[r:r + loc]) + rest
        rec = parse_record(buf, 0, P, ctx.img.encoding, sch.n_cols)
        rec.end = r + loc + 4                       # local footprint on this page
        return rec, buf[:loc], 1
    except (FormatError, UnicodeDecodeError, struct.error, IndexError):
        return None


def _cell_prefix(page: bytes, r: int, P: int, lo: int) -> tuple[int, int] | None:
    """Look back from record start r for 'payload-length varint + rowid varint' that
    ends exactly at r and whose payload length equals P. Returns (cell_start, rowid)."""
    for start in range(max(lo, r - 18), r - 1):
        try:
            p, j = read_varint(page, start, r)
            if p != P:
                continue
            rid, j2 = read_varint(page, j, r)
            if j2 == r:
                return start, rid - (1 << 64) if rid >= (1 << 63) else rid
        except FormatError:
            continue
    return None


def _emit(ctx, sch, rec, raw, ovf, rowid, hyp, region, source, pgno, start, region_end, owner_known,
          payload_ok=math.nan, size_pinned=math.nan, values=None):
    vals = full_row(sch, rec.values if values is None else values, rowid)
    ctx.out.append(Candidate(sch.name, vals, rowid, hyp, region, source, pgno, start, rec.end, region_end,
                             rec.serial_types, owner_known, payload_ok, size_pinned, ovf, raw, ctx.page_id))


def _scan(ctx: _Ctx, page: bytes, lo: int, hi: int, region: str, source: str, pgno: int,
          tables: list[TableSchema], owner_known: bool) -> None:
    """Try 'record' (and the 'cell' prefix in front of it) at every offset of page[lo:hi]."""
    U = ctx.img.usable
    for sch in tables:
        found = 0
        for r in range(lo, hi):
            hb = page[r]
            if hb < sch.n_cols + 1 or (hb > 9 * sch.n_cols + 1 and hb < 0x80):
                continue
            got = _finish_record(ctx, page, r, U, sch)
            if got is None:
                continue
            rec, raw, ovf = got
            P = rec.header_len + sum(serial_len(s) for s in rec.serial_types)
            pre = _cell_prefix(page, r, P, lo)
            if pre is not None:
                _emit(ctx, sch, rec, raw, ovf, pre[1], "cell", region, source, pgno, pre[0], hi, owner_known,
                      payload_ok=1.0)
            _emit(ctx, sch, rec, raw, ovf, None, "record", region, source, pgno, r, hi, owner_known)
            found += 1
            if found >= ctx.cap:
                break


def _freeblock_hypotheses(ctx: _Ctx, page: bytes, s: int, size: int, source: str, pgno: int,
                          sch: TableSchema, owner_known: bool, hyp: str = "freeblock",
                          region: str = "freeblock") -> None:
    """The cell that became this freeblock started at s; bytes s..s+3 are now the
    freeblock header. The record header began at h = s+2 or s+3 (payload-length and
    rowid varints of 1 byte each, plus up to one more byte)."""
    U, enc, n = ctx.img.usable, ctx.img.encoding, sch.n_cols
    for h in (s + 2, s + 3):
        lost_sts = 1 if h == s + 2 else 0              # h=s+3: only the header-length byte is lost
        try:
            sts, k = [], s + 4
            for _ in range(n - lost_sts):
                st, k = read_varint(page, k, U)
                if st in (10, 11):
                    raise FormatError("reserved")
                sts.append(st)
        except FormatError:
            continue
        hlen = k - h
        if hlen < n + 1:
            continue
        options: list[tuple[int, float]] = []          # (first serial type, size_pinned)
        if lost_sts == 0:
            options = [(-1, math.nan)]
        else:
            P_exact = size - (h - s)                   # freeblock covers exactly one cell
            rest = hlen + sum(serial_len(x) for x in sts)
            L0 = P_exact - rest
            first_col = sch.stored[0]
            if sch.ipk == first_col:
                options = [(0, 1.0 if L0 == 0 else 0.0)]
            elif L0 >= 0:
                aff = sch.affinities[first_col]
                if aff == "TEXT":
                    options = [(2 * L0 + 13, 1.0)]
                elif aff == "BLOB":
                    options = [(2 * L0 + 12, 1.0), (2 * L0 + 13, 1.0)]
                else:
                    by_len = {0: [0, 8, 9], 1: [1], 2: [2], 3: [3], 4: [4], 6: [5], 8: [6, 7]}
                    options = [(st, 1.0) for st in by_len.get(L0, [])]
            if not options and sch.affinities[sch.stored[0]] in ("INTEGER", "REAL", "NUMERIC"):
                # merged freeblock: its size no longer fits one cell -> try every number type
                options = [(st, 0.0) for st in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
        for st0, pinned in options:
            full_sts = sts if st0 < 0 else [st0] + sts
            body = k
            P = hlen + sum(serial_len(x) for x in full_sts)
            if local_payload(P, U) != P or h + P > U:
                continue
            try:
                vals = []
                for st in full_sts:
                    vals.append(decode_value(page, body, st, enc))
                    body += serial_len(st)
            except (UnicodeDecodeError, struct.error):
                continue
            if lost_sts == 0:                          # does the freeblock hold exactly this cell?
                pinned = 1.0 if size == (h - s) + P else 0.0
            rec = Record(full_sts, vals, hlen, h, k, body)
            raw = bytes(page[s + 4:body])
            _emit(ctx, sch, rec, raw, 0, None, hyp, region, source, pgno, s, s + size,
                  owner_known, size_pinned=pinned)


def _carve_btree_page(ctx: _Ctx, page: bytes, pgno: int, source: str, sch_list: list[TableSchema],
                      owner_known: bool, old_copy: bool) -> None:
    """Free space of any b-tree page (table or index, leaf or interior: a page that
    changed role keeps old bytes in its free space), plus - for an OLD copy of a
    table leaf - its allocated cells."""
    U = ctx.img.usable
    h = page_header(page, pgno)
    ptr_end = h.base + h.header_size + 2 * h.n_cells
    chain = freeblocks(page, h, U)
    ctx.new_page(cell_pointers(page, h) + [s for s, _ in chain] + [s + n for s, n in chain]
                 + [min(h.content_start, U), U])
    if old_copy and h.page_type == LEAF_TABLE:
        for off in cell_pointers(page, h):
            for sch in sch_list:
                try:
                    P, j = read_varint(page, off, U)
                    rid, j = read_varint(page, j, U)
                except FormatError:
                    continue
                got = _finish_record(ctx, page, j, U, sch)
                if got is not None:
                    rec, raw, ovf = got
                    plen = rec.header_len + sum(serial_len(x) for x in rec.serial_types)
                    _emit(ctx, sch, rec, raw, ovf, rid - (1 << 64) if rid >= (1 << 63) else rid,
                          "old_cell", "cell_area", source, pgno, off, U, owner_known,
                          payload_ok=1.0 if P == plen else 0.0)
    regions = []
    if ptr_end < min(h.content_start, U):
        regions.append(("unallocated", ptr_end, min(h.content_start, U)))
    for s, size in chain:
        regions.append(("freeblock", s + 4, s + size))
        for sch in sch_list:
            _freeblock_hypotheses(ctx, page, s, size, source, pgno, sch, owner_known)
    for region, lo, hi in regions:
        _scan(ctx, page, lo, hi, region, source, pgno, sch_list, owner_known)
        _stale_headers(ctx, page, lo, hi, region, source, pgno, sch_list, owner_known)


def _stale_headers(ctx: _Ctx, page: bytes, lo: int, hi: int, region: str, source: str, pgno: int,
                   sch_list: list[TableSchema], owner_known: bool) -> None:
    """When two neighbouring cells are freed, SQLite merges them into ONE freeblock and
    the inner cell keeps the old 4-byte freeblock header it received when it was freed
    first. Such a stale header = 2-byte 'next' offset (0 or a later offset) + 2-byte
    size (= the old cell size). Every position with that shape inside free space is
    tried as a damaged cell start, with the stale size pinning the lost bytes."""
    U = ctx.img.usable
    for o in range(lo, max(lo, hi - 4)):
        nxt, size = struct.unpack(">HH", page[o:o + 4])
        # the old size may reach past today's free region (space later reused), not past the page
        if size < 8 or o + size > U or not (nxt == 0 or o + size <= nxt < U):
            continue
        ctx.bounds[ctx.page_id].add(o)
        for sch in sch_list:
            _freeblock_hypotheses(ctx, page, o, size, source, pgno, sch, owner_known,
                                  hyp="stale_freeblock", region=region)


def _owner_tables(ctx: _Ctx, pgno: int) -> tuple[list[TableSchema], bool]:
    """Schemas to try on a page: its table's if the page belongs to a user table now,
    else every user table."""
    for name, pages in ctx.img.table_pages.items():
        if name in ctx.tables and pgno in pages:
            return [ctx.tables[name]], True
    return list(ctx.tables.values()), False


def _carve_page_copy(ctx: _Ctx, page: bytes, pgno: int, source: str) -> None:
    """A page that is not a current b-tree page: an old copy, a freelist page or an orphan."""
    sch_list, known = _owner_tables(ctx, pgno)
    try:
        page_header(page, pgno)
        is_btree = True
    except FormatError:
        is_btree = False
    if is_btree:
        _carve_btree_page(ctx, page, pgno, source, sch_list, known, old_copy=True)
    else:                                              # trunk / overflow / zeroed / foreign bytes
        lo = 8 if source == "freelist" else 0
        ctx.new_page([ctx.img.usable])
        _scan(ctx, page, lo, ctx.img.usable, "raw_page", source, pgno, list(ctx.tables.values()), False)


def carve(img: SQLiteImage, cap: int = 400) -> tuple[list[Candidate], dict]:
    """All candidates of one database image, plus counts for the report."""
    tables = {n: s for n, s in img.tables.items() if n != "sqlite_master" and not n.startswith("sqlite_")}
    ctx = _Ctx(img, tables, cap)
    current = sorted(set().union(*[pg for n, pg in img.table_pages.items() if n in tables]) | img.index_pages)
    for pgno in current:
        if pgno in img.pages:
            sch_list, known = _owner_tables(ctx, pgno)
            _carve_btree_page(ctx, img.pages[pgno], pgno, "current", sch_list, known, old_copy=False)
    seen = {hashlib.sha1(img.pages[p]).digest() for p in current if p in img.pages}
    copies = [(p, img.pages[p], "freelist") for p in img.freelist_leaves + img.freelist_trunks if p in img.pages]
    copies += [(p, img.pages[p], "orphan") for p in img.orphan_pages()]
    copies += [(v.pgno, v.data, v.source) for v in img.versions]
    for pgno, data, source in copies:
        dig = hashlib.sha1(data).digest()
        if dig in seen or not any(data):
            continue                                   # identical to a page already carved, or all zero
        seen.add(dig)
        _carve_page_copy(ctx, data, pgno, source)
    return ctx.out, {"candidates": len(ctx.out), "overflow_missing": ctx.n_overflow_missing,
                     "page_copies": len(seen), "bounds": ctx.bounds}


# ---------------------------------------------------------------- grouping + features
def _entropy(b: bytes) -> float:
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _printable(s: str) -> float:
    return sum(ch.isprintable() for ch in s) / len(s) if s else 1.0


_AFF_OK = {"INTEGER": lambda st: st in (0, 1, 2, 3, 4, 5, 6, 8, 9),
           "REAL": lambda st: st in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
           "NUMERIC": lambda st: st in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
           "TEXT": lambda st: st == 0 or (st >= 13 and st % 2 == 1),
           "BLOB": lambda st: True}

FEATURES = (["n_copies", "n_pages", "owner_known", "n_tables_same_width", "rowid_known",
             "rowid_is_live", "rowid_above_live_max", "payload_ok", "size_pinned", "fits_region",
             "overrun", "slack_after", "affinity_match", "null_share", "printable", "text_share",
             "record_len_log", "header_len", "entropy", "zero_share", "live_value_overlap",
             "overflow_followed", "competitors", "end_at_boundary"]
            + [f"hyp_{h}" for h in HYPOTHESES] + [f"src_{s}" for s in SOURCES]
            + [f"reg_{r}" for r in REGIONS])
# The ranker uses STRUCTURE features only (spec-derived). "live_value_overlap" (how many
# of a record's values also occur in live rows) is a CONTENT feature: it is stored for
# the ablation in step 6, which shows it is a shortcut that does not transfer to
# databases built by someone else.
CONTENT_FEATURES = ["live_value_overlap"]
MODEL_FEATURES = [f for f in FEATURES if f not in CONTENT_FEATURES]


@dataclass
class Recovered:
    table: str
    key: tuple
    values: list
    rowid: int | None
    features: dict
    live_duplicate: bool
    best: Candidate


def group_candidates(img: SQLiteImage, cands: list[Candidate], bounds: dict | None = None) -> list[Recovered]:
    """One Recovered record per (table, decoded content). Records whose content equals
    a live row are kept but flagged live_duplicate: they are NEVER reported as deleted."""
    tables = img.tables
    live_keys, live_rowids, live_cols = {}, {}, {}
    for r in img.live:
        sch = tables.get(r.table)
        if sch is None:
            continue
        live_keys.setdefault(r.table, set()).add(row_key(r.values, sch.ipk))
        live_rowids.setdefault(r.table, set()).add(r.rowid)
        for j, v in enumerate(r.values):
            if j != sch.ipk:
                live_cols.setdefault((r.table, j), set()).add(norm_value(v))
    width = Counter(s.n_cols for n, s in tables.items() if not n.startswith("sqlite_"))
    groups: dict[tuple, list[Candidate]] = {}
    for c in cands:
        sch = tables[c.table]
        groups.setdefault((c.table, row_key(c.values, sch.ipk)), []).append(c)
    bounds = {k: set(v) for k, v in (bounds or {}).items()}
    for c in cands:                                   # starts of intact cells are boundaries too
        if c.hypothesis in ("cell", "old_cell"):
            bounds.setdefault(c.page_id, set()).add(c.start)
    spans: dict[tuple, set] = {}
    for g, cs in groups.items():
        for c in cs:
            spans.setdefault((c.source, c.pgno, c.start // 16), set()).add(g)
    out = []
    for (table, key), cs in groups.items():
        sch = tables[table]
        rank = {"old_cell": 0, "cell": 1, "freeblock": 2, "stale_freeblock": 3, "record": 4}
        best = min(cs, key=lambda c: (rank[c.hypothesis], c.end > c.region_end, c.pgno, c.start))
        rowid = next((c.rowid for c in sorted(cs, key=lambda c: rank[c.hypothesis]) if c.rowid is not None), None)
        vals = list(best.values)
        if sch.ipk is not None:
            vals[sch.ipk] = rowid
        texts = [v for v in vals if isinstance(v, str)]
        lmax = max(live_rowids.get(table, {0}) or {0})
        overlap = [norm_value(v) in live_cols.get((table, j), set())
                   for j, v in enumerate(vals) if j != sch.ipk and v is not None]
        comp = set()
        for c in cs:
            comp |= spans.get((c.source, c.pgno, c.start // 16), set())
        f = {
            "n_copies": len(cs), "n_pages": len({(c.source, c.pgno) for c in cs}),
            "owner_known": float(any(c.owner_known for c in cs)),
            "n_tables_same_width": width.get(sch.n_cols, 1),
            "rowid_known": float(rowid is not None),
            "rowid_is_live": float(rowid is not None and rowid in live_rowids.get(table, set())),
            "rowid_above_live_max": float(rowid is not None and rowid > lmax),
            "payload_ok": max((c.payload_ok for c in cs if not math.isnan(c.payload_ok)), default=-1.0),
            "size_pinned": max((c.size_pinned for c in cs if not math.isnan(c.size_pinned)), default=-1.0),
            "fits_region": float(any(c.end == c.region_end for c in cs)),
            "overrun": float(min(max(0, c.end - c.region_end) for c in cs)),
            "slack_after": float(min(max(0, c.region_end - c.end) for c in cs)),
            "affinity_match": sum(_AFF_OK[sch.affinities[col]](st)
                                  for col, st in zip(sch.stored, best.serial_types)) / max(1, sch.n_cols),
            "null_share": sum(st == 0 for st in best.serial_types) / max(1, sch.n_cols),
            "printable": min((_printable(t) for t in texts), default=1.0),
            "text_share": len(texts) / max(1, sch.n_cols),
            "record_len_log": math.log1p(len(best.raw)),
            "header_len": float(best.raw[0]) if best.raw and best.hypothesis != "freeblock" else 0.0,
            "entropy": _entropy(best.raw), "zero_share": best.raw.count(0) / max(1, len(best.raw)),
            "live_value_overlap": sum(overlap) / len(overlap) if overlap else 0.0,
            "overflow_followed": float(any(c.overflow == 1 for c in cs)),
            "competitors": float(len(comp) - 1),
            "end_at_boundary": float(any(c.end in bounds.get(c.page_id, ()) for c in cs)),
        }
        for h in HYPOTHESES:
            f[f"hyp_{h}"] = float(any(c.hypothesis == h for c in cs))
        for s in SOURCES:
            f[f"src_{s}"] = float(any(c.source == s for c in cs))
        for r in REGIONS:
            f[f"reg_{r}"] = float(any(c.region == r for c in cs))
        out.append(Recovered(table, key, vals, rowid, f, key in live_keys.get(table, set()), best))
    return out


def heuristic_score(f: dict) -> float:
    """Structure-only confidence (baseline): share of passed spec checks, in the spirit of
    the fixed heuristic scores of existing tools. Not trained, not calibrated."""
    checks = [f["hyp_old_cell"] or f["hyp_cell"],                  # cell prefix intact
              f["payload_ok"] != 0.0,                              # payload length agrees (or n/a)
              f["affinity_match"] == 1.0,                          # serial types fit the declared types
              f["overrun"] == 0.0,                                 # stays inside its free region
              f["fits_region"] or f["hyp_old_cell"],              # fills its region exactly
              f["printable"] == 1.0,                               # text is printable
              f["size_pinned"] != 0.0]                             # freeblock size agrees (or n/a)
    return sum(map(float, checks)) / len(checks)


def is_strict(f: dict) -> bool:
    """Classic structural carving: report only records whose cell is fully intact and
    consistent (no ML, no inference of lost bytes)."""
    return bool((f["hyp_old_cell"] or f["hyp_cell"]) and f["payload_ok"] == 1.0
                and f["affinity_match"] == 1.0 and f["overrun"] == 0.0)
