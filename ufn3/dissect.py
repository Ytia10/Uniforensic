"""External baseline: DC3 SQLite Dissect 1.0.0 (pip package sqlite-dissect, Python 3).

Run as the tool's own command-line program with carving switched on:
  -c  carve deleted cells   -k  carve freelist pages   -e csv  CSV output
It finds a database's WAL / journal in the same folder by itself. A row counts as
"reported deleted" when the tool marks its Operation as Carved or Deleted. Its CSV
holds every value as text, so values are compared in a canonical text form
(integers as integers, reals to 5 decimals, the rest as text).
"""
from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

META = ["File Source", "Version", "Page Version", "Cell Source", "Page Number", "Location", "Operation",
        "File Offset", "Row ID"]
REPORTED = {"Carved", "Deleted"}
_CLI = ("import sys; from sqlite_dissect.entrypoint import cli; "
        "sys.argv[0] = 'sqlite_dissect'; sys.exit(cli())")


def canon_text(s):
    """Canonical form of a value written as text (CSV cell or answer-key entry)."""
    if s is None or s == "" or s == "NULL" or s == "None":
        return ("null",)
    try:
        return ("num", int(s))
    except (ValueError, TypeError):
        pass
    try:
        f = float(s)
        if math.isfinite(f):
            return ("num", int(f)) if f == int(f) and abs(f) < 2 ** 53 else ("num", round(f, 5))
    except (ValueError, TypeError):
        pass
    return ("text", str(s))


def canon_value(v):
    """Canonical form of a typed value (from our reader or from SQLite)."""
    if v is None:
        return ("null",)
    if isinstance(v, (bytes, bytearray)):
        return ("text", bytes(v).hex())
    return canon_text(v if isinstance(v, str) else repr(v) if isinstance(v, float) else str(v))


def run(db_path: Path, timeout: int = 180) -> tuple[list[tuple[str, list[str]]], float, str]:
    """Rows the tool reports as deleted: [(table, values as text)], seconds, error text
    ("" when the tool finished normally). Rows written before a crash are kept."""
    with tempfile.TemporaryDirectory(prefix="dissect_") as out:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-c", _CLI, str(db_path), "-d", out, "-e", "csv", "-c", "-k"],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            p = None
        secs = time.time() - t0
        rows = []
        prefix = db_path.name + "-"
        for f in sorted(Path(out).glob("*.csv")):
            table = f.stem[len(prefix):] if f.stem.startswith(prefix) else f.stem
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                rd = csv.reader(fh)
                head = next(rd, None)
                if not head or head[:len(META)] != META:
                    continue
                op = head.index("Operation")
                for r in rd:
                    if len(r) >= len(META) and r[op] in REPORTED:
                        rows.append((table, r[len(META):]))
        if p is None:
            err = f"timeout after {timeout}s"
        elif p.returncode == 0:
            err = ""
        else:
            lines = [ln for ln in p.stderr.splitlines() if "Error" in ln or "Exception" in ln]
            err = (lines[-1] if lines else f"exit code {p.returncode}")[:200]
        return rows, secs, err
