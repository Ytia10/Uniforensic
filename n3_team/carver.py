"""
carver.py
Structure-first SQLite record carver and spec-derived feature extractor.
Scans freeblocks (including 4-byte overwrite reconstruction and coalesced cells),
unallocated space, and freelist leaf pages.
Extracts candidate records and computes 8 explainable, spec-derived features.
"""

import struct
import math
import os
from typing import List, Dict, Any, Tuple, Optional, Set
import sqlite_spec


class SQLiteCarver:
    def __init__(self, schema_info: Dict[str, Any]):
        """
        schema_info:
        {
            "table_name": str,
            "columns": List[str],
            "column_types": List[str]  # e.g. ["INTEGER", "TEXT", "TEXT", "TEXT", "INTEGER", "INTEGER"]
        }
        """
        self.schema_info = schema_info
        self.table_name = schema_info["table_name"]
        self.columns = schema_info["columns"]
        self.column_types = [t.upper() for t in schema_info.get("column_types", [])]
        self.expected_col_count = len(self.columns)

    def scan_database(self, db_path: str) -> List[Dict[str, Any]]:
        """
        Scans an SQLite database file for candidate records across:
        1. Freeblock chains in leaf pages (including 4-byte overwrite reconstruction)
        2. Coalesced deleted cells within freeblocks
        3. Unallocated space
        4. Freelist pages
        """
        if not os.path.exists(db_path):
            return []

        with open(db_path, "rb") as f:
            data = f.read()

        if len(data) < 512 or data[:16] != b"SQLite format 3\x00":
            return []

        page_size = int.from_bytes(data[16:18], "big")
        if page_size == 1:
            page_size = 65536

        num_pages = len(data) // page_size
        candidates = []

        # 1. Parse Freelist pages
        first_freelist_page = int.from_bytes(data[32:36], "big")
        freelist_page_count = int.from_bytes(data[36:40], "big")
        freelist_pages = self._get_freelist_pages(data, page_size, first_freelist_page, freelist_page_count)

        # 2. Iterate pages
        for page_idx in range(num_pages):
            page_offset = page_idx * page_size
            is_page_1 = (page_idx == 0)
            btree_hdr_offset = page_offset + (100 if is_page_1 else 0)

            if page_idx + 1 in freelist_pages:
                # Carve freelist leaf page
                cands = self._carve_raw_region(
                    data,
                    start_offset=page_offset,
                    length=page_size,
                    source_type="freelist",
                    page_idx=page_idx + 1
                )
                candidates.extend(cands)
                continue

            page_type = data[btree_hdr_offset]
            if page_type != sqlite_spec.PAGE_TYPE_TABLE_LEAF:
                continue

            first_freeblock = int.from_bytes(data[btree_hdr_offset + 1 : btree_hdr_offset + 3], "big")
            cell_count = int.from_bytes(data[btree_hdr_offset + 3 : btree_hdr_offset + 5], "big")
            cell_content_start = int.from_bytes(data[btree_hdr_offset + 5 : btree_hdr_offset + 7], "big")
            if cell_content_start == 0:
                cell_content_start = 65536

            cell_ptr_array_start = btree_hdr_offset + 8
            cell_ptr_array_end = cell_ptr_array_start + 2 * cell_count

            # A. Scan unallocated space
            if cell_content_start > (cell_ptr_array_end - page_offset):
                unallocated_start = cell_ptr_array_end
                unallocated_len = (page_offset + cell_content_start) - unallocated_start
                if unallocated_len > 12:
                    cands = self._carve_raw_region(
                        data,
                        start_offset=unallocated_start,
                        length=unallocated_len,
                        source_type="unallocated",
                        page_idx=page_idx + 1
                    )
                    candidates.extend(cands)

            # B. Traverse Freeblock chain
            fb_offset = first_freeblock
            visited_fb = set()
            while fb_offset != 0 and fb_offset not in visited_fb and fb_offset + 4 <= page_size:
                visited_fb.add(fb_offset)
                abs_fb_offset = page_offset + fb_offset
                next_fb = int.from_bytes(data[abs_fb_offset : abs_fb_offset + 2], "big")
                fb_size = int.from_bytes(data[abs_fb_offset + 2 : abs_fb_offset + 4], "big")

                if fb_size < 4 or fb_offset + fb_size > page_size:
                    break

                fb_cands = self._carve_freeblock(
                    data,
                    abs_offset=abs_fb_offset,
                    size=fb_size,
                    page_idx=page_idx + 1
                )
                candidates.extend(fb_cands)
                fb_offset = next_fb

        return self._deduplicate_candidates(candidates)

    def _get_freelist_pages(self, data: bytes, page_size: int, trunk_page: int, total_pages: int) -> Set[int]:
        freelist_leaf_pages = set()
        curr_trunk = trunk_page
        visited_trunks = set()

        while curr_trunk != 0 and curr_trunk not in visited_trunks:
            visited_trunks.add(curr_trunk)
            trunk_offset = (curr_trunk - 1) * page_size
            if trunk_offset + 8 > len(data):
                break
            next_trunk = int.from_bytes(data[trunk_offset : trunk_offset + 4], "big")
            leaf_count = int.from_bytes(data[trunk_offset + 4 : trunk_offset + 8], "big")

            for i in range(leaf_count):
                ptr_offset = trunk_offset + 8 + i * 4
                if ptr_offset + 4 > len(data):
                    break
                leaf_p = int.from_bytes(data[ptr_offset : ptr_offset + 4], "big")
                freelist_leaf_pages.add(leaf_p)

            curr_trunk = next_trunk

        return freelist_leaf_pages

    def _carve_freeblock(
        self,
        data: bytes,
        abs_offset: int,
        size: int,
        page_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Carves candidate records from a freeblock:
        1. Checks offset 4 where serial types start (reconstructing overwritten header)
        2. Scans internal offsets to find coalesced deleted records
        """
        candidates = []
        fb_bytes = data[abs_offset : abs_offset + size]

        # Target number of payload columns (excluding AUTOINCREMENT primary key if not stored in body)
        # In SQLite, table with INTEGER PRIMARY KEY usually has (N - 1) payload columns
        target_cols = [self.expected_col_count - 1, self.expected_col_count]

        # Scan all potential starting offsets within the freeblock
        for off in range(size - 12):
            is_fb_boundary = (off == 4)

            # Strategy A: Direct record parse (intact header)
            vals, st, tb, ok = sqlite_spec.parse_record(fb_bytes, off, size - off)
            if ok and len(st) in target_cols:
                feats = self._extract_features(
                    fb_bytes, off, tb, size - off, vals, st, "freeblock", is_fb_boundary
                )
                candidates.append({
                    "abs_offset": abs_offset + off,
                    "page_idx": page_idx,
                    "source_type": "freeblock",
                    "values": vals,
                    "serial_types": st,
                    "total_bytes": tb,
                    "features": feats,
                    "strategy": "direct"
                })
                continue

            # Strategy B: Reconstructed serial-type parse (where header length was overwritten by freeblock 4-byte header)
            for num_c in target_cols:
                serials = []
                curr = off
                valid_chain = True
                for _ in range(num_c):
                    st_val, n = sqlite_spec.read_varint(fb_bytes, curr)
                    if st_val is None or n == 0 or st_val in (10, 11):
                        valid_chain = False
                        break
                    serials.append(st_val)
                    curr += n

                if valid_chain and len(serials) == num_c:
                    # Check if serial types match schema roughly
                    val_off = curr
                    vals_decoded = []
                    fit = True
                    for s in serials:
                        sz = sqlite_spec.serial_type_size(s)
                        if sz is None or val_off + sz > size:
                            fit = False
                            break
                        v = sqlite_spec.decode_serial_value(s, fb_bytes[val_off : val_off + sz])
                        vals_decoded.append(v)
                        val_off += sz

                    if fit and any(isinstance(v, str) and len(v) >= 3 for v in vals_decoded):
                        tb_reconstructed = val_off - off
                        feats = self._extract_features(
                            fb_bytes, off, tb_reconstructed, size - off, vals_decoded, serials, "freeblock", is_fb_boundary
                        )
                        candidates.append({
                            "abs_offset": abs_offset + off,
                            "page_idx": page_idx,
                            "source_type": "freeblock",
                            "values": vals_decoded,
                            "serial_types": serials,
                            "total_bytes": tb_reconstructed,
                            "features": feats,
                            "strategy": "reconstructed"
                        })
                        break

        return candidates

    def _carve_raw_region(
        self,
        data: bytes,
        start_offset: int,
        length: int,
        source_type: str,
        page_idx: int
    ) -> List[Dict[str, Any]]:
        candidates = []
        target_cols = [self.expected_col_count - 1, self.expected_col_count]

        for off in range(length - 12):
            vals, st, tb, ok = sqlite_spec.parse_record(data, start_offset + off, length - off)
            if ok and len(st) in target_cols:
                feats = self._extract_features(
                    data, start_offset + off, tb, length - off, vals, st, source_type, False
                )
                candidates.append({
                    "abs_offset": start_offset + off,
                    "page_idx": page_idx,
                    "source_type": source_type,
                    "values": vals,
                    "serial_types": st,
                    "total_bytes": tb,
                    "features": feats,
                    "strategy": "raw"
                })
        return candidates

    def _extract_features(
        self,
        data: bytes,
        offset: int,
        total_bytes: int,
        avail_len: int,
        values: List[Any],
        serial_types: List[int],
        source_type: str,
        is_fb_boundary: bool
    ) -> Dict[str, float]:
        """Extracts 8 explainable features."""
        # 1. Varint validity
        f1_varint_valid = 1.0

        # 2. Serial type legality
        f2_serial_legal = 1.0 if all(st not in (10, 11) for st in serial_types) else 0.0

        # 3. Schema type match
        matches = 0
        total_check = min(len(serial_types), len(self.column_types))
        offset_idx = 1 if len(self.column_types) == len(serial_types) + 1 else 0

        for i in range(len(serial_types)):
            expected_t = self.column_types[i + offset_idx] if (i + offset_idx) < len(self.column_types) else ""
            st = serial_types[i]
            if "TEXT" in expected_t and sqlite_spec.is_text_type(st):
                matches += 1
            elif "INT" in expected_t and (sqlite_spec.is_integer_type(st) or st == sqlite_spec.SERIAL_NULL):
                matches += 1
            elif "BLOB" in expected_t and sqlite_spec.is_blob_type(st):
                matches += 1
        f3_schema_match = matches / max(1, total_check)

        # 4. Length fit
        f4_length_fit = min(1.0, total_bytes / max(1, avail_len)) if total_bytes <= avail_len else 0.0

        # 5. Text validity
        printable_chars = 0
        total_chars = 0
        for val in values:
            if isinstance(val, str):
                total_chars += len(val)
                printable_chars += sum(1 for c in val if c.isprintable() or c in "\r\n\t")
        f5_text_validity = (printable_chars / total_chars) if total_chars > 0 else 0.7

        # 6. Value sanity (timestamps, counts, text format)
        sanity_score = 0
        sanity_tests = 0
        for val in values:
            if isinstance(val, int):
                sanity_tests += 1
                if (1262304000 <= val <= 2051222400) or (1262304000000 <= val <= 2051222400000):
                    sanity_score += 1.0  # Plausible epoch timestamp
                elif 0 <= val <= 100000:
                    sanity_score += 1.0  # Plausible count / duration / status
                else:
                    sanity_score += 0.2
            elif isinstance(val, str):
                sanity_tests += 1
                if len(val) >= 2:
                    sanity_score += 1.0
        f6_value_sanity = (sanity_score / sanity_tests) if sanity_tests > 0 else 0.5

        # 7. Byte entropy
        raw_slice = data[offset : offset + min(total_bytes, len(data) - offset)]
        entropy = sqlite_spec.calculate_entropy(raw_slice)
        if 2.5 <= entropy <= 6.5:
            f7_entropy_score = 1.0
        elif entropy < 2.5:
            f7_entropy_score = max(0.0, entropy / 2.5)
        else:
            f7_entropy_score = max(0.0, (8.0 - entropy) / 1.5)

        # 8. Freeblock boundary alignment
        f8_freeblock_alignment = 1.0 if is_fb_boundary else (0.8 if source_type == "freeblock" else 0.5)

        return {
            "f1_varint_valid": float(f1_varint_valid),
            "f2_serial_legal": float(f2_serial_legal),
            "f3_schema_match": float(f3_schema_match),
            "f4_length_fit": float(f4_length_fit),
            "f5_text_validity": float(f5_text_validity),
            "f6_value_sanity": float(f6_value_sanity),
            "f7_entropy_score": float(f7_entropy_score),
            "f8_freeblock_alignment": float(f8_freeblock_alignment)
        }

    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates candidates having identical values."""
        seen = {}
        for cand in candidates:
            # Create a hashable key from stringified values
            key = tuple(str(v) for v in cand["values"])
            if key not in seen:
                seen[key] = cand
            else:
                # Keep the candidate with higher feature scores
                old_cand = seen[key]
                old_sum = sum(old_cand["features"].values())
                new_sum = sum(cand["features"].values())
                if new_sum > old_sum:
                    seen[key] = cand
        return list(seen.values())
