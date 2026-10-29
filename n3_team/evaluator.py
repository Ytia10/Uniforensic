"""
evaluator.py
Comprehensive Forensic Benchmark Evaluator for SQLite Record Carving (Novelty 3).
Compares:
1. Baseline 1: Regex / Raw String Carver (bulk search without SQLite structure)
2. Baseline 2: Spec-Only Structural Carver (bring2lite / Undark style heuristic without ML or live-row exclusion)
3. Proposed N3 System: Structural Carver + Spec-Derived ML Classifier + Platt Calibration + Live-Row Exclusion Filter

Evaluates:
- Recall of deleted rows
- Precision (with Live-Row False Positives counted separately from Spurious False Positives)
- Live-Row False Positive Count & Rate
- Expected Calibration Error (ECE) and Brier Score
Across schemas (SMS, Calls, Browser History) and PRAGMAs (secure_delete=OFF, FAST, ON; journal_mode=WAL).
"""

import sqlite3
import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import sqlite_spec
from database_generator import ForensicDatabaseGenerator
from carver import SQLiteCarver
from classifier import CalibratedSQLiteClassifier


def get_live_rows(db_path: str, table_name: str) -> List[Dict[str, Any]]:
    """Fetches all active (live) rows from the SQLite database."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table_name};")
        col_names = [desc[0] for desc in cur.description]
        rows = []
        for r in cur.fetchall():
            rows.append(dict(zip(col_names, r)))
        conn.close()
        return rows
    except Exception:
        conn.close()
        return []


def is_row_match(candidate_values: List[Any], target_row: Dict[str, Any]) -> bool:
    """Checks if candidate decoded values match a ground truth row."""
    # target_row has columns e.g. ["id", "sender", "recipient", "body", "timestamp", "status"]
    # candidate_values may not have "id" (AUTOINCREMENT primary key)
    target_vals = list(target_row.values())
    cand_str = " ".join(str(v) for v in candidate_values if v is not None)
    
    # Check 1: Exact slice match (ignoring id)
    if candidate_values == target_vals[1:]:
        return True
    if candidate_values == target_vals:
        return True

    # Check 2: Core fields match (e.g. sender and recipient and body)
    matched_core = 0
    total_core = 0
    for k, v in target_row.items():
        if k in ("id",):
            continue
        total_core += 1
        if str(v) in cand_str:
            matched_core += 1

    return (matched_core >= total_core - 1) and (matched_core >= 2)


# Baseline 1: Regex / Bulk String Carver
class RegexBaselineCarver:
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info

    def carve(self, db_path: str) -> List[Dict[str, Any]]:
        with open(db_path, "rb") as f:
            data = f.read()

        candidates = []
        # Find phone numbers or text patterns
        phone_pattern = rb"(\+1-555-\d{4}|BankNotify|2FA-Service|SupportDesk)"
        matches = [m.start() for m in re.finditer(phone_pattern, data)]

        for pos in matches:
            # Extract a pseudo-record of 64 bytes
            chunk = data[pos : pos + 80]
            # Try splitting printable strings
            text_parts = [p.decode("utf-8", errors="ignore") for p in chunk.split(b"\x00") if len(p) >= 3]
            if len(text_parts) >= 2:
                candidates.append({
                    "values": text_parts,
                    "confidence": 0.5,
                    "abs_offset": pos
                })

        return candidates


# Baseline 2: Spec-Only Structural Carver (bring2lite / Undark style)
class SpecOnlyStructuralCarver:
    def __init__(self, schema_info: Dict[str, Any]):
        self.carver = SQLiteCarver(schema_info)

    def carve(self, db_path: str) -> List[Dict[str, Any]]:
        candidates = self.carver.scan_database(db_path)
        # Spec-only: outputs all candidates without ML ranking or calibrated confidence
        # Sets uncalibrated confidence = 1.0 for all parsed records
        for c in candidates:
            c["confidence"] = 1.0
        return candidates


# Proposed N3 System: Structural Carver + Calibrated Classifier + Live-Row Exclusion
class N3ProposedCarver:
    def __init__(self, schema_info: Dict[str, Any], classifier: CalibratedSQLiteClassifier, threshold: float = 0.5):
        self.carver = SQLiteCarver(schema_info)
        self.classifier = classifier
        self.threshold = threshold
        self.table_name = schema_info["table_name"]

    def carve(self, db_path: str) -> List[Dict[str, Any]]:
        # 1. Structural Candidate Generation
        raw_candidates = self.carver.scan_database(db_path)
        if not raw_candidates:
            return []

        # 2. Spec-Derived ML Scoring & Platt Calibration
        feature_dicts = [c["features"] for c in raw_candidates]
        probs = self.classifier.predict_proba(feature_dicts)

        for c, p in zip(raw_candidates, probs):
            c["confidence"] = float(p)

        # 3. Live-Row Exclusion by Construction
        live_rows = get_live_rows(db_path, self.table_name)
        filtered_candidates = []

        for c in raw_candidates:
            is_live = False
            for lr in live_rows:
                if is_row_match(c["values"], lr):
                    is_live = True
                    break
            c["is_live_row"] = is_live
            # Only report if NOT a live row and meets calibrated confidence threshold
            if not is_live and c["confidence"] >= self.threshold:
                filtered_candidates.append(c)

        return filtered_candidates


def evaluate_carver(
    carver_name: str,
    candidates: List[Dict[str, Any]],
    deleted_rows: List[Dict[str, Any]],
    live_rows: List[Dict[str, Any]],
    apply_live_filter: bool = False
) -> Dict[str, Any]:
    """
    Evaluates carving results against ground truth.
    Distinguishes:
    - True Positives (TP): matched deleted row
    - Live-Row False Positives (FP_live): matched an active live row (major failure mode!)
    - Spurious False Positives (FP_spurious): corrupted garbage / noise
    """
    total_deleted = len(deleted_rows)
    total_live = len(live_rows)

    matched_deleted = set()
    fp_live_count = 0
    fp_spurious_count = 0

    for cand in candidates:
        vals = cand.get("values", [])

        # Check if matches a deleted row
        matched_del_idx = None
        for i, d in enumerate(deleted_rows):
            if i not in matched_deleted and is_row_match(vals, d):
                matched_del_idx = i
                break

        if matched_del_idx is not None:
            matched_deleted.add(matched_del_idx)
        else:
            # Check if it matches a live row
            is_live = False
            for lr in live_rows:
                if is_row_match(vals, lr):
                    is_live = True
                    break

            if is_live:
                if not apply_live_filter:
                    fp_live_count += 1
            else:
                fp_spurious_count += 1

    tp_count = len(matched_deleted)
    total_reported = tp_count + fp_live_count + fp_spurious_count

    recall = (tp_count / total_deleted) if total_deleted > 0 else 0.0
    precision = (tp_count / total_reported) if total_reported > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "carver_name": carver_name,
        "total_deleted": total_deleted,
        "recovered_tp": tp_count,
        "fp_live": fp_live_count,
        "fp_spurious": fp_spurious_count,
        "total_reported": total_reported,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1)
    }
