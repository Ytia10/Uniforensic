"""One case version end to end: plan deletions -> build the 5 SQLite databases ->
carve them with the teammate's N3 -> label every recovered record against the truth.

Labels of a recovered record (a carved record that is a valid row and is not an
exact copy of a live row):
  correct - exactly equals a row the SUSPECT deleted (evidence, distractor or background)
  foreign - exactly equals a deleted row that belongs to someone else (foreign residue)
  wrong   - a valid-looking row that never existed (fragments, overlaps, overwritten
            cells decoded into plausible values)
The truth labels are used ONLY for evaluation; the triage pipeline sees p alone.
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd

from .devicedb import TABLES, build_db
from .n3_adapter import carve
from .scenario import plan


def run_case(case_id: str, tables: dict[str, pd.DataFrame], level: float, foreign: bool,
             clf, scfg: dict, seed: int, window_days: int, keep_dir: str | None = None):
    """Return (truth DataFrame, recovered DataFrame) for one case version."""
    events, truth = plan(case_id, tables, level, foreign, scfg, seed, window_days)
    lookup = {}                                   # deleted stored row -> truth record
    for r in truth[truth.deleted].itertuples(index=False):
        lookup.setdefault((r.table, r.row), r)
    rec_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in TABLES:
            if not events[t]:
                continue
            db = os.path.join(keep_dir or tmp, f"{case_id}_{t}.db")
            build_db(db, t, events[t])
            for c in carve(db, t, clf):
                if c["artifact"] is None:          # not a valid row of this table: unusable
                    continue
                hit = lookup.get((t, c["values"]))
                label = "wrong" if hit is None else ("foreign" if hit.owner == "foreign" else "correct")
                rec_rows.append({"case_id": case_id, "table": t, **c["artifact"],
                                 "p_raw": c["p_raw"], "fuzzy_live": bool(c["fuzzy_live"]),
                                 "strategy": c["strategy"], "source_type": c["source_type"],
                                 "label": label,
                                 "match_id": "" if hit is None else hit.artifact_id,
                                 "match_role": "" if hit is None else hit.role,
                                 "match_pattern": "" if hit is None else hit.pattern_id})
    truth = truth.drop(columns=["row"])
    truth["case_id"] = case_id
    return truth, pd.DataFrame(rec_rows)
