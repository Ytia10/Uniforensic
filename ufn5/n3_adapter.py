"""Use the teammate's N3 code (folder n3_team/) WITHOUT changing a single line of it.

What is used from N3, and how:
  carver.SQLiteCarver               - structure-first record carving (freeblocks,
                                      unallocated space, freelist pages) + 8 features
  classifier.CalibratedSQLiteClassifier - logistic regression + Platt scaling -> p
  database_generator.ForensicDatabaseGenerator, evaluator.is_row_match
                                    - to train the classifier exactly as N3's Step 1-2

N3's training step lives inside run_n3_pipeline.run_pipeline(), so it cannot be
imported on its own. train_teammate_classifier() repeats those lines VERBATIM, with
one documented addition: numpy's global random generator is seeded first, because
N3 draws its synthetic negative feature vectors from it without a seed (so every N3
run gives a slightly different model). n5_01 proves the copy is exact: it runs the
teammate's own run_pipeline() with the same seed and checks that the weights and
Platt parameters are bit-for-bit identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ufn1.common import ROOT

N3_DIR = ROOT / "n3_team"
N3_FILES = {                       # md5 of the files the teammate shared (checked by n5_00)
    "carver.py": "a1272eb16400db2a81ce060707197e68",
    "classifier.py": "ad656958c157a6f477e84a695a6d28a5",
    "database_generator.py": "861d1baa1cd3a39da49f2cd8203f709a",
    "evaluator.py": "5a26c276304af3633d1a25f80d096ae5",
    "generate_diagrams.py": "0d39e1f340c9bc6fadfa82c3d5a73c34",
    "run_n3_pipeline.py": "339d6b7d33686ef8bf1c3e46b5ed64eb",
    "sqlite_spec.py": "771f07f193f73948d77a87616757a2f3",
}
if str(N3_DIR) not in sys.path:   # the N3 modules import each other as top-level modules
    sys.path.insert(0, str(N3_DIR))

from carver import SQLiteCarver                       # noqa: E402  (teammate's code)
from classifier import CalibratedSQLiteClassifier     # noqa: E402
from database_generator import ForensicDatabaseGenerator  # noqa: E402
from evaluator import is_row_match                    # noqa: E402

from .devicedb import from_row, normalise, read_live_rows, schema_info, SCHEMAS  # noqa: E402

# ---- copied verbatim from run_n3_pipeline.py (Step 1 / Step 2) ----------------
TRAIN_CONFIGS = [
    ("train_sms_1", "sms", 50, 20, 0, 0, "DELETE", 101),
    ("train_sms_2", "sms", 60, 25, 0, 0, "DELETE", 102),
    ("train_calls_1", "calls", 50, 20, 0, 0, "DELETE", 103),
    ("train_browser_1", "browser", 50, 20, 0, 0, "DELETE", 104),
]
N3_SCHEMA_DEFS = {
    "sms": {"table_name": "messages",
            "columns": ["id", "sender", "recipient", "body", "timestamp", "status"],
            "column_types": ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER", "INTEGER"]},
    "calls": {"table_name": "calls",
              "columns": ["id", "phone_number", "duration_sec", "timestamp", "call_type"],
              "column_types": ["INTEGER", "TEXT", "INTEGER", "INTEGER", "TEXT"]},
    "browser": {"table_name": "browser_history",
                "columns": ["id", "url", "title", "visit_time", "visit_count"],
                "column_types": ["INTEGER", "TEXT", "TEXT", "INTEGER", "INTEGER"]},
}


def train_teammate_classifier(db_dir: Path, numpy_seed: int) -> tuple[CalibratedSQLiteClassifier, dict]:
    """N3 Step 1 + Step 2, line for line, after np.random.seed(numpy_seed)."""
    np.random.seed(numpy_seed)                          # the ONE addition (see module doc)
    generator = ForensicDatabaseGenerator(output_dir=str(db_dir))
    train_X, train_y = [], []
    for name, stype, n_rows, n_del, sec_del, auto_vac, j_mode, seed in TRAIN_CONFIGS:
        db_path, gt_path = generator.create_database(
            name, schema_type=stype, num_rows=n_rows, num_delete=n_del,
            secure_delete=sec_del, auto_vacuum=auto_vac, journal_mode=j_mode, seed=seed)
        with open(gt_path) as f:
            gt = json.load(f)
        candidates = SQLiteCarver(N3_SCHEMA_DEFS[stype]).scan_database(db_path)
        deleted_rows = gt["deleted_rows"]
        for c in candidates:
            train_X.append(c["features"])
            train_y.append(1 if any(is_row_match(c["values"], d) for d in deleted_rows) else 0)
        for _ in range(len(candidates) // 2 + 5):       # N3's injected negative distractors
            train_X.append({
                "f1_varint_valid": float(np.random.choice([0.0, 1.0], p=[0.7, 0.3])),
                "f2_serial_legal": float(np.random.choice([0.0, 1.0], p=[0.4, 0.6])),
                "f3_schema_match": float(np.random.uniform(0.0, 0.4)),
                "f4_length_fit": float(np.random.uniform(0.1, 0.5)),
                "f5_text_validity": float(np.random.uniform(0.1, 0.6)),
                "f6_value_sanity": float(np.random.uniform(0.0, 0.4)),
                "f7_entropy_score": float(np.random.uniform(0.1, 0.5)),
                "f8_freeblock_alignment": float(np.random.choice([0.5, 0.8]))})
            train_y.append(0)
    classifier = CalibratedSQLiteClassifier(l2_reg=0.01)
    classifier.fit(train_X, train_y, val_split=0.25)
    probs_train = classifier.predict_proba(train_X)
    calib = classifier.evaluate_calibration(probs_train, np.array(train_y), num_bins=10)
    info = {"n_train_vectors": len(train_X), "n_positive": int(sum(train_y)),
            "ece_in_sample": calib["ece"], "brier_in_sample": calib["brier_score"]}
    return classifier, info


def classifier_to_json(clf: CalibratedSQLiteClassifier) -> dict:
    return {"l2_reg": clf.l2_reg, "weights": [float(w) for w in clf.weights],
            "bias": float(clf.bias), "platt_a": clf.platt_a, "platt_b": clf.platt_b,
            "feature_names": clf.FEATURE_NAMES}


def classifier_from_json(d: dict) -> CalibratedSQLiteClassifier:
    clf = CalibratedSQLiteClassifier(l2_reg=d["l2_reg"])
    clf.weights, clf.bias = np.array(d["weights"]), d["bias"]
    clf.platt_a, clf.platt_b, clf.is_fitted = d["platt_a"], d["platt_b"], True
    return clf


# ------------------------------------------------------------- carving
def carve(db_path: str, table: str, clf: CalibratedSQLiteClassifier) -> list[dict]:
    """Run the teammate's carver + classifier on one database.

    Returns one dict per DISTINCT carved record that is NOT an exact copy of a live
    row. Keys: values (stored columns, no rowid), artifact (unified-schema fields or
    None if the record is not a valid row), p_raw (teammate's calibrated p),
    fuzzy_live (True if the teammate's own live-row filter, evaluator.is_row_match,
    would have dropped it), strategy, source_type.

    Live-row exclusion here is EXACT (all stored columns equal a live row). The
    teammate's filter is fuzzy (all-but-one columns as substrings) and also drops
    genuinely deleted rows that resemble a live row; it is recorded, not applied,
    so its effect can be measured (n5_03)."""
    cands = SQLiteCarver(schema_info(table)).scan_database(db_path)
    if not cands:
        return []
    probs = clf.predict_proba([c["features"] for c in cands])
    live = read_live_rows(db_path, table)
    cols = SCHEMAS[table]["columns"]
    live_dicts = [dict(zip(cols, (0,) + r)) for r in live]   # same shape as N3's get_live_rows
    best: dict[tuple, dict] = {}
    for c, p in zip(cands, probs):
        v = normalise(table, c["values"])
        if v is None or v in live:
            continue
        art = from_row(table, v)
        if v in best and best[v]["p_raw"] >= float(p):
            continue
        best[v] = {"values": v, "artifact": art, "p_raw": float(p),
                   "fuzzy_live": None, "strategy": c["strategy"], "source_type": c["source_type"]}
    for r in best.values():               # fuzzy check only where it can matter (valid rows)
        if r["artifact"] is not None:
            r["fuzzy_live"] = any(is_row_match(list(r["values"]), lr) for lr in live_dicts)
    return list(best.values())
