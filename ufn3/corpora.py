"""Third-party test corpora: where to get them, how they are verified, their truth.

1. SQLite Forensic Corpus v2.0 (Nemetz, Schmitt & Freiling, DFRWS EU 2018), CC0.
   Official download: https://downloads.digitalcorpora.org/corpora/sql/sqlite_forensic_corpus_v2.0.zip
2. NIST CFReDS "SQLite Data Recovery (deleted and modified records)", test SFT-03,
   U.S. Government work (public domain). Portal:
   https://cfreds.nist.gov/all/NIST/SQLiteDataRecoverydeletedandmodifiedrecords
Both are also kept, unchanged, in the open-source sqlite4n6 repository
(https://github.com/SecurityRonin/sqlite-forensic, folder tests/data/), which is the
easiest way to get them. Wherever the files come from, every one is checked against
the MD5 list in corpora_md5.json (the NIST files against the MD5s NIST publishes), so
the numbers can only be produced on the exact same bytes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name("corpora_md5.json").read_text())

# NIST SFT-03 answer key, copied from NIST's SQLiteDocumentation (CFReDS SFT-03):
# 100 rows deleted by InvoiceLineId, and 100 rows modified (Quantity set to 200).
SFT03_DELETED = [int(x) for x in (
    "253;1285;1154;1413;515;1794;1947;367;1693;274;855;1510;1451;1092;545;2188;1783;1590;131;1396;"
    "2144;1781;1098;201;263;981;2003;1727;922;347;2227;1274;365;329;521;772;2145;1374;1457;1253;366;"
    "1538;1390;615;1219;1653;1256;340;1982;1356;1943;1708;1532;1468;817;833;959;167;97;1902;1097;697;"
    "830;1927;381;136;2136;2239;2084;1585;1883;43;738;1410;525;1810;1591;598;2025;631;357;828;172;"
    "1864;1732;717;197;1131;1710;2115;1283;1225;963;1609;1076;1099;1201;1717;1546;961").split(";")]
SFT03_MODIFIED = [int(x) for x in (
    "1804;2035;2189;1687;1421;885;678;681;1662;1684;679;1694;61;2157;1088;1587;1861;226;1903;27;1234;"
    "1353;1523;1138;150;294;1365;2194;1121;906;1592;2206;771;2118;2058;1614;1107;239;1246;1293;470;701;"
    "536;1210;5;2039;2132;199;543;1194;1275;1702;1628;889;1358;44;127;943;1072;1366;1208;1307;120;46;"
    "2087;951;1728;555;1796;213;277;1081;147;1928;1715;2032;546;1049;92;526;1017;624;468;1805;1544;"
    "1286;328;861;880;2122;1818;733;1964;386;552;599;1370;1057;907;2174").split(";")]
SFT03_VARIANTS = {   # database file -> (sidecar file, sidecar kind)
    "SFT-03_PERSIST_android.sqlite": ("SFT-03_PERSIST_android.sqlite-journal", "journal"),
    "SFT-03_PERSIST_ios.sqlite": ("SFT-03_PERSIST_ios.sqlite-journal", "journal"),
    "sft-03-WAL_android.sqlite": ("sft-03-WAL_android.sqlite-wal", "wal"),
    "sft-03-WAL_ios.sqlite": ("sft-03-WAL_ios.sqlite-wal", "wal"),
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect(sources: list[Path], dest: Path) -> tuple[int, list[str]]:
    """Copy every manifest file found anywhere under `sources` into dest/<corpus>/<name>,
    matching by MD5 (so file and folder names do not matter). Returns (found, missing)."""
    want = {md5: rel for rel, md5 in MANIFEST.items()}
    found = {rel for rel, md5 in MANIFEST.items() if (dest / rel).exists() and _md5(dest / rel) == md5}
    for src in sources:
        for f in src.rglob("*"):
            if len(found) == len(MANIFEST):
                break
            if not f.is_file() or f.stat().st_size > 5_000_000:
                continue
            rel = want.get(_md5(f))
            if rel and rel not in found:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dest / rel)
                found.add(rel)
    return len(found), sorted(set(MANIFEST) - found)


def verify(dest: Path) -> list[str]:
    """Problems with the copied corpora (empty list = all files present and unchanged)."""
    return [rel for rel, md5 in MANIFEST.items() if not (dest / rel).exists() or _md5(dest / rel) != md5]


# ------------------------------------------------------------------ truth per database
def _typed(aff: str, s):
    """Answer-key text -> the value SQLite stored, using the column's affinity."""
    if s is None or s == "NULL":
        return None
    if aff in ("INTEGER", "NUMERIC", "REAL"):
        try:
            return float(s) if aff == "REAL" else int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    return s


def nemetz_truth(img, xml_path: Path, blobs: list[bytes]) -> dict:
    """table -> deleted / live / present keys from the Nemetz XML answer key."""
    from .dissect import canon_text
    from .nemetz import answer_key
    from .truth import record_body
    out = {}
    for table, t in answer_key(xml_path).items():
        sch = img.tables.get(table)
        if sch is None:
            continue
        ipk = sch.ipk
        key = lambda vals: tuple(canon_text(v) for j, v in enumerate(vals) if j != ipk)  # noqa: E731
        present = set()
        for vals in t["deleted"]:
            body = record_body([_typed(a, v) for a, v in zip(sch.affinities, vals)], ipk)
            if any(body in b for b in blobs):
                present.add(key(vals))
        out[table] = {"ipk": ipk, "deleted": {key(v) for v in t["deleted"]}, "live": {key(v) for v in t["live"]},
                      "present": present, "neutral": set()}
    return out


def _sqlite_rows(db: Path, sidecar: Path | None, table: str) -> dict:
    """Rows of `table` as SQLite itself returns them, read from COPIES (never the evidence)."""
    import tempfile

    import apsw
    with tempfile.TemporaryDirectory(prefix="ufn3_ro_") as tmp:
        dst = Path(tmp) / db.name
        shutil.copyfile(db, dst)
        if sidecar is not None:
            shutil.copyfile(sidecar, Path(tmp) / sidecar.name)
        con = apsw.Connection(str(dst), flags=apsw.SQLITE_OPEN_READONLY)
        rows = {r[0]: r for r in con.execute(f'SELECT * FROM "{table}"')}
        con.close()
    return rows


def cfreds_truth(folder: Path, name: str, blobs: list[bytes]) -> dict:
    """invoice_items truth for one SFT-03 variant. Deleted rows: NIST's list; their
    content is read by SQLite from the WAL variant's main file of the same platform,
    which still holds every row (its WAL was never checkpointed). Rows NIST modified
    (Quantity = 200): an older copy of such a row is a correct recovery of a PRIOR
    VERSION, so it is scored as neutral (neither right nor wrong)."""
    from .dissect import canon_value
    from .truth import record_body
    platform = "android" if "android" in name else "ios"
    before = _sqlite_rows(folder / f"sft-03-WAL_{platform}.sqlite", None, "invoice_items")
    side, _kind = SFT03_VARIANTS[name]
    now = _sqlite_rows(folder / name, folder / side if side.endswith("-wal") else None, "invoice_items")
    assert set(before) - set(now) == set(SFT03_DELETED), "SFT-03 files do not match NIST's deleted list"
    key = lambda r: tuple(canon_value(v) for v in r[1:])                       # noqa: E731 (drop the rowid alias)
    deleted = {key(before[i]) for i in SFT03_DELETED}
    present = {key(before[i]) for i in SFT03_DELETED
               if any(record_body(list(before[i]), 0) in b for b in blobs)}
    neutral = {key(now[i])[:3] for i in SFT03_MODIFIED}                         # same row, other Quantity
    return {"invoice_items": {"ipk": 0, "deleted": deleted, "live": {key(r) for r in now.values()},
                              "present": present, "neutral": neutral}}
