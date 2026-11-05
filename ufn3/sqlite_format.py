"""A read-only reader for the SQLite file format (main file + optional WAL file).

Written from the official specification ("Database File Format",
https://www.sqlite.org/fileformat.html). The facts this module relies on:

  * The file is a sequence of pages; page 1 starts with a 100-byte header
    (page size at offset 16, reserved bytes per page at 20, change counter at 24,
    page count at 28, first freelist trunk page at 32, freelist page count at 36,
    auto-vacuum root at 52, text encoding at 56, version-valid-for at 92).
  * A b-tree page header is 8 bytes (leaf) or 12 bytes (interior): page type
    (2/5/10/13 = interior index / interior table / leaf index / leaf table), first
    freeblock, cell count, start of the cell-content area (0 means 65536),
    fragmented free bytes, and for interior pages the right-most child.
  * A table-leaf cell = payload length (varint) + rowid (varint) + payload, plus a
    4-byte first-overflow-page number when the payload does not fit locally.
  * A record = header length (varint) + one serial type (varint) per column + the
    values. Serial types: 0 NULL, 1-6 big-endian ints of 1,2,3,4,6,8 bytes,
    7 float64, 8 integer 0, 9 integer 1, 10/11 reserved (never valid),
    even N>=12 BLOB of (N-12)/2 bytes, odd N>=13 TEXT of (N-13)/2 bytes.
  * A freeblock = 2-byte offset of the next freeblock + 2-byte size, so freeing a
    cell OVERWRITES ITS FIRST 4 BYTES - the reason deleted records are often damaged.
  * Freelist trunk page = next trunk (4 bytes) + leaf count L (4 bytes) + L leaf
    page numbers; freelist leaf pages are not written by SQLite, so old content
    can survive on them.
  * WAL file = 32-byte header (magic 0x377f0682/3, version, page size, checkpoint
    sequence, two salts, two checksums) then frames of 24-byte header + one page.
    A frame is valid when its salts match the header and its running checksum
    matches; a frame with a non-zero "database size" field is a commit frame.

The reader never writes and never calls the SQLite library, so it sees exactly the
bytes an examiner would acquire.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .sqlite_schema import TableSchema, parse_create_table

LEAF_TABLE, INTERIOR_TABLE, LEAF_INDEX, INTERIOR_INDEX = 13, 5, 10, 2
BTREE_TYPES = (LEAF_TABLE, INTERIOR_TABLE, LEAF_INDEX, INTERIOR_INDEX)
WAL_MAGIC = (0x377F0682, 0x377F0683)
ENCODINGS = {1: "utf-8", 2: "utf-16-le", 3: "utf-16-be"}
_INT_SIZES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 8}


class FormatError(ValueError):
    """Raised when bytes do not follow the SQLite format."""


# ------------------------------------------------------------------ primitives
def read_varint(b: bytes, i: int, end: int) -> tuple[int, int]:
    """SQLite varint at b[i:]: 1-9 bytes, big-endian, high bit = 'more bytes follow'
    for the first 8 bytes; the 9th byte contributes all 8 bits. Returns (value, next_i)."""
    v = 0
    for k in range(8):
        if i + k >= end:
            raise FormatError("varint runs past the end")
        c = b[i + k]
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i + k + 1
    if i + 8 >= end:
        raise FormatError("varint runs past the end")
    v = (v << 8) | b[i + 8]
    return v & 0xFFFFFFFFFFFFFFFF, i + 9


def varint_len(v: int) -> int:
    if v < 0:
        v &= 0xFFFFFFFFFFFFFFFF
    if v >= (1 << 56):
        return 9
    n = 1
    while v >= (1 << (7 * n)):
        n += 1
    return n


def encode_varint(v: int) -> bytes:
    """Inverse of read_varint (used by the self-test and the reconstruction code)."""
    v &= 0xFFFFFFFFFFFFFFFF
    if v >= (1 << 56):
        out = [v & 0xFF]
        v >>= 8
        for _ in range(8):
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(out))
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def serial_len(st: int) -> int:
    """Number of body bytes for a serial type (-1 for the reserved types 10 and 11)."""
    if st >= 12:
        return (st - 12) // 2
    if st in _INT_SIZES:
        return _INT_SIZES[st]
    if st == 7:
        return 8
    if st in (0, 8, 9):
        return 0
    return -1


def decode_value(b: bytes, i: int, st: int, enc: str):
    """Value of serial type st stored at b[i:]. Text is decoded strictly; a failure
    raises UnicodeDecodeError, which the carver turns into a feature."""
    if st == 0:
        return None
    if st == 8:
        return 0
    if st == 9:
        return 1
    if st in _INT_SIZES:
        n = _INT_SIZES[st]
        return int.from_bytes(b[i:i + n], "big", signed=True)
    if st == 7:
        return struct.unpack(">d", b[i:i + 8])[0]
    n = (st - 12) // 2
    raw = bytes(b[i:i + n])
    return raw if st % 2 == 0 else raw.decode(enc)


@dataclass
class Record:
    serial_types: list[int]
    values: list
    header_len: int
    start: int          # offset of the header-length varint
    body_start: int
    end: int            # offset just after the last value


def parse_record(b: bytes, i: int, end: int, enc: str, n_cols: int | None = None) -> Record:
    """Strict record parse at b[i:end]. Raises FormatError / UnicodeDecodeError."""
    hlen, j = read_varint(b, i, end)
    hend = i + hlen
    if hlen < 1 or hend > end or hlen > 9 * 2000:
        raise FormatError("bad header length")
    sts = []
    while j < hend:
        st, j = read_varint(b, j, hend)
        if st in (10, 11):
            raise FormatError("reserved serial type")
        sts.append(st)
    if j != hend:
        raise FormatError("header length does not match serial types")
    if n_cols is not None and len(sts) != n_cols:
        raise FormatError("column count mismatch")
    k, vals = hend, []
    for st in sts:
        n = serial_len(st)
        if k + n > end:
            raise FormatError("value runs past the end")
        vals.append(decode_value(b, k, st, enc))
        k += n
    return Record(sts, vals, hlen, i, hend, k)


# ------------------------------------------------------------------ page helpers
@dataclass
class PageHeader:
    page_type: int
    first_freeblock: int
    n_cells: int
    content_start: int
    fragmented: int
    right_child: int
    header_size: int
    base: int           # 100 on page 1, else 0


def page_header(page: bytes, pgno: int) -> PageHeader:
    base = 100 if pgno == 1 else 0
    t = page[base]
    if t not in BTREE_TYPES:
        raise FormatError(f"page {pgno}: not a b-tree page (type byte {t})")
    fb, nc, cs, fr = struct.unpack(">HHHB", page[base + 1:base + 8])
    cs = 65536 if cs == 0 else cs
    interior = t in (INTERIOR_TABLE, INTERIOR_INDEX)
    rc = struct.unpack(">I", page[base + 8:base + 12])[0] if interior else 0
    return PageHeader(t, fb, nc, cs, fr, rc, 12 if interior else 8, base)


def cell_pointers(page: bytes, h: PageHeader) -> list[int]:
    o = h.base + h.header_size
    return list(struct.unpack(f">{h.n_cells}H", page[o:o + 2 * h.n_cells]))


def freeblocks(page: bytes, h: PageHeader, usable: int) -> list[tuple[int, int]]:
    """(offset, size) of every freeblock on the chain. Stops at loops or bad links."""
    out, seen, off = [], set(), h.first_freeblock
    while off and off not in seen and off + 4 <= usable:
        seen.add(off)
        nxt, size = struct.unpack(">HH", page[off:off + 4])
        if size < 4 or off + size > usable:
            break
        out.append((off, size))
        off = nxt
    return out


def local_payload(P: int, usable: int, table_leaf: bool = True) -> int:
    """Bytes of a payload of size P stored on the b-tree page itself (spec, 'Cell
    Payload Overflow Pages'): X = U-35 for table leaves, ((U-12)*64/255)-23 for
    index pages; M = ((U-12)*32/255)-23; K = M+((P-M) % (U-4))."""
    X = usable - 35 if table_leaf else ((usable - 12) * 64 // 255) - 23
    if P <= X:
        return P
    M = ((usable - 12) * 32 // 255) - 23
    K = M + ((P - M) % (usable - 4))
    return K if K <= X else M


# ------------------------------------------------------------------ the image
@dataclass
class Version:
    """One copy of a page that is NOT the current one (still holds old data)."""
    pgno: int
    data: bytes
    source: str          # superseded_main | superseded_wal | wal_uncommitted | wal_stale | beyond_eof | journal


@dataclass
class LiveRow:
    table: str
    rowid: int
    values: list
    page: int


@dataclass
class SQLiteImage:
    page_size: int
    reserved: int
    encoding: str
    n_pages: int
    pages: dict[int, bytes]                     # current version of every page
    versions: list[Version]                     # older copies (main file + WAL)
    tables: dict[str, TableSchema]
    table_pages: dict[str, set[int]]            # leaf + interior pages of each table b-tree
    leaf_owner: dict[int, str]                  # leaf page -> table
    index_pages: set[int]
    overflow_pages: set[int]
    freelist_trunks: list[int]
    freelist_leaves: list[int]
    ptrmap_pages: set[int]
    live: list[LiveRow]
    autovacuum: bool
    wal_info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> int:
        return self.page_size - self.reserved

    def orphan_pages(self) -> list[int]:
        used = {1} | self.index_pages | self.overflow_pages | self.ptrmap_pages
        used |= set(self.freelist_trunks) | set(self.freelist_leaves)
        for s in self.table_pages.values():
            used |= s
        return [p for p in range(1, self.n_pages + 1) if p not in used and p in self.pages]


def _wal_checksum(data: bytes, s0: int, s1: int, big: bool) -> tuple[int, int]:
    fmt = ">" if big else "<"
    words = struct.unpack(f"{fmt}{len(data) // 4}I", data)
    for k in range(0, len(words), 2):
        s0 = (s0 + words[k] + s1) & 0xFFFFFFFF
        s1 = (s1 + words[k + 1] + s0) & 0xFFFFFFFF
    return s0, s1


def parse_wal(wal: bytes, page_size: int):
    """Split WAL frames into committed / uncommitted / stale. Returns
    (committed list of (pgno, data, db_size), other list of (pgno, data, kind), info)."""
    info = {"frames": 0, "committed": 0, "uncommitted": 0, "stale": 0}
    if len(wal) < 32:
        return [], [], info
    magic, ver, ps, seq, salt1, salt2, c1, c2 = struct.unpack(">8I", wal[:32])
    if magic not in WAL_MAGIC:
        raise FormatError("bad WAL magic")
    big = bool(magic & 1)
    ps = ps or page_size
    s0, s1 = _wal_checksum(wal[:24], 0, 0, big)
    header_ok = (s0, s1) == (c1, c2)
    fsize = 24 + ps
    n = (len(wal) - 32) // fsize
    info["frames"] = n
    committed, pending, other, valid_chain = [], [], [], header_ok
    for k in range(n):
        off = 32 + k * fsize
        pgno, dbsize, fs1, fs2, fc1, fc2 = struct.unpack(">6I", wal[off:off + 24])
        data = wal[off + 24:off + fsize]
        if valid_chain and (fs1, fs2) == (salt1, salt2):
            s0, s1 = _wal_checksum(wal[off:off + 8] + data, s0, s1, big)
            if (s0, s1) == (fc1, fc2):
                pending.append((pgno, data, dbsize))
                if dbsize:
                    committed.extend(pending)
                    pending = []
                continue
            valid_chain = False
        other.append((pgno, data, "wal_stale"))
    other = [(p, d, "wal_uncommitted") for p, d, _ in pending] + other
    info.update(committed=len(committed), uncommitted=len(pending),
                stale=len(other) - len(pending), header_ok=header_ok)
    return committed, other, info


def parse_journal(journal: bytes, page_size: int, max_page: int) -> list[tuple[int, bytes]]:
    """Page images kept in a rollback journal (-journal file). Each record is a 4-byte
    page number, one page and a 4-byte checksum; records follow a header that fills one
    sector. In PERSIST mode the header is zeroed after commit, so the sector size is
    unknown: the scan starts at the first 512-byte boundary (the smallest sector size)
    and, after a record that does not look valid, moves on to the next boundary."""
    out, rec, pos = [], 4 + page_size + 4, 512
    while pos + rec <= len(journal):
        pgno = struct.unpack(">I", journal[pos:pos + 4])[0]
        page = journal[pos + 4:pos + 4 + page_size]
        if 1 <= pgno <= max_page and any(page):
            out.append((pgno, page))
            pos += rec
        else:
            pos = (pos // 512 + 1) * 512
    return out


def open_image(db: bytes, wal: bytes | None = None, journal: bytes | None = None) -> SQLiteImage:
    """Parse a database image (and its WAL / rollback journal, if any) into pages,
    schema and live rows. Older page copies go to `versions`."""
    if len(db) < 100 or db[:16] != b"SQLite format 3\x00":
        raise FormatError("not an SQLite database")
    ps = struct.unpack(">H", db[16:18])[0]
    ps = 65536 if ps == 1 else ps
    reserved = db[20]
    enc = ENCODINGS.get(struct.unpack(">I", db[56:60])[0], "utf-8")
    hdr_pages = struct.unpack(">I", db[28:32])[0]
    valid_for, counter = struct.unpack(">I", db[92:96])[0], struct.unpack(">I", db[24:28])[0]
    file_pages = len(db) // ps
    n_pages = hdr_pages if (hdr_pages and valid_for == counter) else file_pages
    pages = {p: db[(p - 1) * ps:p * ps] for p in range(1, file_pages + 1)}
    versions: list[Version] = []
    wal_info: dict = {}
    if wal:
        committed, other, wal_info = parse_wal(wal, ps)
        from_wal: set[int] = set()
        for pgno, data, dbsize in committed:          # newest committed frame of a page wins
            if pgno in pages:
                versions.append(Version(pgno, pages[pgno],
                                        "superseded_wal" if pgno in from_wal else "superseded_main"))
            pages[pgno] = data
            from_wal.add(pgno)
            if dbsize:
                n_pages = dbsize
        versions += [Version(p, d, kind) for p, d, kind in other]
    if journal:
        versions += [Version(p, d, "journal") for p, d in parse_journal(journal, ps, max(file_pages, n_pages) + 64)]
    for p in [p for p in pages if p > n_pages]:            # truncated by a commit / vacuum
        versions.append(Version(p, pages.pop(p), "beyond_eof"))
    img = SQLiteImage(ps, reserved, enc, n_pages, pages, versions, {}, {}, {}, set(), set(),
                      [], [], set(), [], autovacuum=struct.unpack(">I", db[52:56])[0] != 0,
                      wal_info=wal_info)
    # if page 1 was rewritten in the WAL, header fields come from the newest copy
    p1 = pages[1]
    if p1[:16] == b"SQLite format 3\x00":
        img.encoding = ENCODINGS.get(struct.unpack(">I", p1[56:60])[0], enc)
        img.autovacuum = struct.unpack(">I", p1[52:56])[0] != 0
    _read_freelist(img, struct.unpack(">I", p1[32:36])[0])
    if img.autovacuum:
        _mark_ptrmap(img)
    _read_schema(img)
    return img


def _mark_ptrmap(img: SQLiteImage) -> None:
    """Pointer-map pages (auto_vacuum databases): page 2, then every U/5 + 1 pages."""
    step = img.usable // 5 + 1
    p = 2
    while p <= img.n_pages:
        img.ptrmap_pages.add(p)
        p += step


def _read_freelist(img: SQLiteImage, trunk: int) -> None:
    seen = set()
    while trunk and trunk not in seen and trunk in img.pages:
        seen.add(trunk)
        pg = img.pages[trunk]
        nxt, cnt = struct.unpack(">II", pg[:8])
        cnt = min(cnt, (img.usable - 8) // 4)
        img.freelist_trunks.append(trunk)
        img.freelist_leaves += [x for x in struct.unpack(f">{cnt}I", pg[8:8 + 4 * cnt])
                                if 0 < x <= img.n_pages]
        trunk = nxt


def read_overflow(img_pages: dict[int, bytes], first: int, need: int, usable: int) -> tuple[bytes, list[int]]:
    """Follow an overflow chain (4-byte next pointer + data per page)."""
    out, chain, p = bytearray(), [], first
    while p and len(out) < need and p in img_pages and p not in chain:
        pg = img_pages[p]
        chain.append(p)
        out += pg[4:usable]
        p = struct.unpack(">I", pg[:4])[0]
    if len(out) < need:
        raise FormatError("overflow chain broken")
    return bytes(out[:need]), chain


def table_leaf_cells(img: SQLiteImage, pgno: int, page: bytes):
    """Yield (offset, rowid, payload_bytes, overflow_chain) for every cell of a leaf
    table page, following overflow chains in the CURRENT image."""
    h = page_header(page, pgno)
    U = img.usable
    for off in cell_pointers(page, h):
        P, j = read_varint(page, off, U)
        rowid, j = read_varint(page, j, U)
        loc = local_payload(P, U)
        payload, chain = page[j:j + loc], []
        if loc < P:
            first = struct.unpack(">I", page[j + loc:j + loc + 4])[0]
            rest, chain = read_overflow(img.pages, first, P - loc, U)
            payload = payload + rest
        yield off, rowid - (1 << 64) if rowid >= (1 << 63) else rowid, payload, chain


def walk_btree(img: SQLiteImage, root: int, on_leaf=None) -> set[int]:
    """All pages of the b-tree rooted at `root`; calls on_leaf(pgno, page) for leaves."""
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p in seen or p not in img.pages:
            continue
        seen.add(p)
        page = img.pages[p]
        h = page_header(page, p)
        if h.page_type in (INTERIOR_TABLE, INTERIOR_INDEX):
            for off in cell_pointers(page, h):
                stack.append(struct.unpack(">I", page[off:off + 4])[0])
            stack.append(h.right_child)
        elif on_leaf:
            on_leaf(p, page, h)
    return seen


def _read_schema(img: SQLiteImage) -> None:
    master = TableSchema("sqlite_master", ["type", "name", "tbl_name", "rootpage", "sql"],
                         ["text"] * 5, ["TEXT"] * 5, None, root_page=1, stored=list(range(5)))
    rows = []

    def collect(p, page, h):
        if h.page_type != LEAF_TABLE:
            return
        for off, rowid, payload, chain in table_leaf_cells(img, p, page):
            img.overflow_pages.update(chain)
            rows.append(parse_record(payload, 0, len(payload), img.encoding).values)

    img.table_pages["sqlite_master"] = walk_btree(img, 1, collect)
    img.tables["sqlite_master"] = master
    for typ, name, _tbl, root, sql in rows:
        if not root:
            continue
        if typ == "table":
            sch = parse_create_table(name, sql or "", root)
            if sch is None or sch.without_rowid:
                img.warnings.append(f"table {name}: not carved (virtual / WITHOUT ROWID / unreadable)")
                img.index_pages |= _safe_walk(img, root)    # still mark its pages as used
                continue
            img.tables[name] = sch
            img.table_pages[name] = set()
        else:                                               # index
            img.index_pages |= _safe_walk(img, root)
    for name, sch in img.tables.items():
        if name == "sqlite_master":
            continue

        def collect_live(p, page, h, name=name, sch=sch):
            if h.page_type != LEAF_TABLE:
                return
            img.leaf_owner[p] = name
            for off, rowid, payload, chain in table_leaf_cells(img, p, page):
                img.overflow_pages.update(chain)
                rec = parse_record(payload, 0, len(payload), img.encoding)
                img.live.append(LiveRow(name, rowid, full_row(sch, rec.values, rowid), p))

        img.table_pages[name] = walk_btree(img, sch.root_page, collect_live)
    for p in img.table_pages["sqlite_master"]:
        img.leaf_owner.setdefault(p, "sqlite_master")


def _safe_walk(img, root) -> set[int]:
    try:
        return walk_btree(img, root)
    except FormatError:
        return set()


def full_row(sch: TableSchema, values: list, rowid: int | None) -> list:
    """Record values -> one value per declared column (rowid put into the alias column;
    columns missing from a short record - added by ALTER TABLE - become None)."""
    vals = list(values) + [None] * (len(sch.stored) - len(values))
    row = [None] * len(sch.columns)
    for j, v in zip(sch.stored, vals):
        # SQLite may store an integral REAL as an integer on disk and converts it
        # back to a float when it reads the column; do the same
        row[j] = float(v) if sch.affinities[j] == "REAL" and isinstance(v, int) else v
    if sch.ipk is not None:
        row[sch.ipk] = rowid
    return row
