"""Run the alternative N3 (package ufn3, unchanged) on one N5 device database, and return
its recovered records in EXACTLY the format N5 uses for the teammate's carver
(ufn5.n3_adapter.carve), so every later N5 step runs unchanged.

What is used from ufn3, and how:
  sqlite_format.open_image  - raw-byte reader of the main file + -wal / -journal (never writes)
  carver.carve              - the 5 hypotheses over every place deleted data can survive
  carver.group_candidates   - one recovered record per decoded content, 41 spec features
  models_n3/ranker.joblib   - the gradient-boosted ranker + isotonic calibration that
                              n3_03_train_ranker.py trained on N3's own databases (built
                              from N1 TRAIN cases) and calibrated on N1 VALIDATION cases
Records whose content equals a live row are dropped exactly as N5 does it (ufn5.n3_adapter):
by equality of all stored columns. `fuzzy_live` records what the teammate's fuzzy live-row
rule (n3_team/evaluator.is_row_match) WOULD drop; it is recorded, not applied, so N5's
variant Af keeps its meaning.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ufn1.common import ROOT, load_json
from ufn3.carver import carve as ufn3_carve, group_candidates
from ufn3.sqlite_format import FormatError, open_image
from ufn5.devicedb import SCHEMAS, from_row, read_live_rows
from ufn5.n3_adapter import is_row_match          # the teammate's matching rule (for variant Af only)


def load_ranker(cfg5b: dict) -> dict:
    """The trained + calibrated ranker of the alternative N3 (written by n3_03_train_ranker.py)."""
    path = ROOT / cfg5b["carver"]["ranker"]
    meta = load_json(path.with_suffix(".json"))
    bundle = joblib.load(path)
    return {"model": bundle, "features": bundle["features"], "calibration": meta["calibration"],
            "cap": cfg5b["carver"]["max_candidates_per_region"]}


def carve(db_path: str, table: str, ranker: dict) -> list[dict]:
    """Same contract as ufn5.n3_adapter.carve: one dict per DISTINCT recovered record that is
    not an exact copy of a live row. Keys: values (stored columns, no rowid), artifact
    (unified-schema fields, or None if not a valid row), p_raw (ufn3's calibrated p),
    fuzzy_live, strategy (ufn3 hypothesis), source_type (where the bytes were found)."""
    p = Path(db_path)
    side = [p.with_name(p.name + s) for s in ("-wal", "-journal")]
    wal, jrn = (s.read_bytes() if s.exists() else None for s in side)
    try:
        img = open_image(p.read_bytes(), wal, jrn)
    except FormatError:
        return []
    cands, info = ufn3_carve(img, ranker["cap"])
    name = SCHEMAS[table]["table_name"]
    groups = [g for g in group_candidates(img, cands, info["bounds"])
              if g.table == name and not g.live_duplicate]
    if not groups:
        return []
    X = np.array([[g.features[f] for f in ranker["features"]] for g in groups], dtype=float)
    probs = ranker["model"]["calibrators"][ranker["calibration"]].predict_proba(X)[:, 1]
    ipk = img.tables[name].ipk
    n_stored = len(SCHEMAS[table]["columns"]) - 1
    live = read_live_rows(db_path, table)
    cols = SCHEMAS[table]["columns"]
    live_dicts = [dict(zip(cols, (0,) + r)) for r in live]   # same shape as N3's get_live_rows
    best: dict[tuple, dict] = {}
    for g, pr in zip(groups, probs):
        v = tuple(x for j, x in enumerate(g.values) if j != ipk)   # drop the rowid alias column
        if len(v) != n_stored or v in live:
            continue
        if v in best and best[v]["p_raw"] >= float(pr):
            continue
        best[v] = {"values": v, "artifact": from_row(table, v), "p_raw": float(pr), "fuzzy_live": None,
                   "strategy": g.best.hypothesis, "source_type": g.best.source}
    for r in best.values():               # fuzzy check only where it can matter (valid rows)
        if r["artifact"] is not None:
            r["fuzzy_live"] = any(is_row_match(list(r["values"]), lr) for lr in live_dicts)
    return list(best.values())
