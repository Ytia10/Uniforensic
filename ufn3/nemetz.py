"""The SQLite Forensic Corpus (Nemetz, Schmitt & Freiling, DFRWS EU 2018), version 2.0.

Third-party databases with an XML answer key per database: every row of every table
with deleted="1" for deleted rows. N3 never trains on it; it is the test of whether
a ranker trained on N3's synthetic databases still works on databases built by
someone else, with other schemas, another SQLite version (3.20.1) and deliberate
anti-forensic manipulation (categories 17, 18).

Values in the answer key are text; they are compared with recovered values through
dissect.canon_text / canon_value (integers as integers, reals to the 5 decimals the
key uses, text exactly; the key writes NULL as the text 'NULL').
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def find_databases(folder: Path, categories: list[str]) -> list[tuple[str, Path, Path]]:
    """(id, db path, xml path) for every answer key in the chosen categories, wherever
    it sits under `folder` (works for the official zip and for other copies)."""
    out = []
    for x in sorted(folder.rglob("*.xml")):
        stem = x.stem
        if stem[:2] not in categories:
            continue
        for cand in (x.with_suffix(".db"), x.with_name(stem + "_antifor.db")):
            if cand.exists():
                out.append((stem, cand, x))
                break
    return out


def answer_key(xml_path: Path) -> dict[str, dict]:
    """table -> {"deleted": [row, ...], "live": [row, ...]}; a row is a list of strings."""
    root = ET.parse(xml_path).getroot()
    out = {}
    for el in root.findall("element"):
        meta = el.find("meta")
        if meta is None or meta.findtext("type") != "table":
            continue
        rows = el.find("entries")
        t = {"deleted": [], "live": [], "dropped": meta.findtext("deleted") == "True"}
        for r in (rows.findall("row") if rows is not None else []):
            vals = [c.findtext("content") for c in r.findall("column")]
            t["deleted" if r.get("deleted") == "1" else "live"].append(vals)
        out[meta.findtext("name")] = t
    return out
