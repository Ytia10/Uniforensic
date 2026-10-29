"""
sqlite_spec.py
Implementation of the official SQLite Database File Format Specification (sqlite.org/fileformat2.html).
Handles varint decoding, serial type interpretation, record header parsing, and B-tree page navigation.
"""

import struct
import math
from typing import List, Tuple, Optional, Dict, Any

# SQLite Serial Type Constants
SERIAL_NULL = 0
SERIAL_INT8 = 1
SERIAL_INT16 = 2
SERIAL_INT24 = 3
SERIAL_INT32 = 4
SERIAL_INT48 = 5
SERIAL_INT64 = 6
SERIAL_FLOAT64 = 7
SERIAL_CONST_ZERO = 8
SERIAL_CONST_ONE = 9
SERIAL_RESERVED_10 = 10
SERIAL_RESERVED_11 = 11

# B-Tree Page Types
PAGE_TYPE_INDEX_INTERIOR = 0x02
PAGE_TYPE_TABLE_INTERIOR = 0x05
PAGE_TYPE_INDEX_LEAF = 0x0A
PAGE_TYPE_TABLE_LEAF = 0x0D


def read_varint(data: bytes, offset: int = 0) -> Tuple[Optional[int], int]:
    """Reads a variable-length integer (varint) from bytes starting at offset."""
    val = 0
    for i in range(8):
        if offset + i >= len(data):
            return None, 0
        b = data[offset + i]
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, i + 1
    # 9th byte uses all 8 bits
    if offset + 8 < len(data):
        b = data[offset + 8]
        val = (val << 8) | b
        return val, 9
    return None, 0


def encode_varint(n: int) -> bytes:
    """Encodes a non-negative integer into an SQLite varint."""
    if n < 0:
        raise ValueError("Varint must be non-negative")
    if n <= 0x7F:
        return bytes([n])
    parts = []
    if n >= (1 << 56):
        parts.append(n & 0xFF)
        n >>= 8
        while n > 0:
            parts.append((n & 0x7F) | 0x80)
            n >>= 7
        while len(parts) < 9:
            parts.append(0x80)
        return bytes(reversed(parts))

    while n > 0:
        parts.append(n & 0x7F)
        n >>= 7
    parts.reverse()
    out = [p | 0x80 for p in parts[:-1]]
    out.append(parts[-1])
    return bytes(out)


def serial_type_size(st: int) -> Optional[int]:
    """Returns the size in bytes of the payload for a given serial type."""
    if st == SERIAL_NULL: return 0
    if st == SERIAL_INT8: return 1
    if st == SERIAL_INT16: return 2
    if st == SERIAL_INT24: return 3
    if st == SERIAL_INT32: return 4
    if st == SERIAL_INT48: return 6
    if st == SERIAL_INT64: return 8
    if st == SERIAL_FLOAT64: return 8
    if st == SERIAL_CONST_ZERO: return 0
    if st == SERIAL_CONST_ONE: return 0
    if st in (SERIAL_RESERVED_10, SERIAL_RESERVED_11):
        return None  # Illegal in well-formed file
    if st >= 12 and (st % 2 == 0):
        return (st - 12) // 2  # BLOB
    if st >= 13 and (st % 2 == 1):
        return (st - 13) // 2  # TEXT
    return None


def is_text_type(st: int) -> bool:
    return st >= 13 and (st % 2 == 1)


def is_blob_type(st: int) -> bool:
    return st >= 12 and (st % 2 == 0)


def is_integer_type(st: int) -> bool:
    return st in (1, 2, 3, 4, 5, 6, 8, 9)


def decode_serial_value(st: int, data: bytes) -> Any:
    """Decodes a single column value from bytes using its serial type."""
    size = serial_type_size(st)
    if size is None or len(data) < size:
        return None
    if st == SERIAL_NULL: return None
    if st == SERIAL_INT8: return struct.unpack(">b", data[:1])[0]
    if st == SERIAL_INT16: return struct.unpack(">h", data[:2])[0]
    if st == SERIAL_INT24:
        b = data[:3]
        prefix = b'\xff' if (b[0] & 0x80) else b'\x00'
        return struct.unpack(">i", prefix + b)[0]
    if st == SERIAL_INT32: return struct.unpack(">i", data[:4])[0]
    if st == SERIAL_INT48:
        b = data[:6]
        prefix = b'\xff\xff' if (b[0] & 0x80) else b'\x00\x00'
        return struct.unpack(">q", prefix + b)[0]
    if st == SERIAL_INT64: return struct.unpack(">q", data[:8])[0]
    if st == SERIAL_FLOAT64: return struct.unpack(">d", data[:8])[0]
    if st == SERIAL_CONST_ZERO: return 0
    if st == SERIAL_CONST_ONE: return 1
    if is_blob_type(st):
        return bytes(data[:size])
    if is_text_type(st):
        try:
            return data[:size].decode("utf-8")
        except UnicodeDecodeError:
            return data[:size].decode("utf-8", errors="replace")
    return None


def parse_record_header(data: bytes, offset: int = 0, max_header_len: int = 128) -> Tuple[Optional[List[int]], int, bool]:
    """
    Parses an SQLite record header starting at offset.
    Returns (serial_types, total_header_len, is_valid).
    """
    if offset >= len(data):
        return None, 0, False

    header_len, varint_bytes = read_varint(data, offset)
    if header_len is None or varint_bytes == 0:
        return None, 0, False
    if header_len < varint_bytes or header_len > min(max_header_len, len(data) - offset):
        return None, 0, False

    curr_offset = offset + varint_bytes
    header_end = offset + header_len
    serial_types = []

    while curr_offset < header_end:
        st, st_bytes = read_varint(data, curr_offset)
        if st is None or st_bytes == 0:
            return None, 0, False
        if st in (10, 11):  # Reserved, strictly illegal
            return None, 0, False
        serial_types.append(st)
        curr_offset += st_bytes

    if curr_offset != header_end:
        return None, 0, False

    return serial_types, header_len, True


def parse_record(data: bytes, offset: int = 0, max_len: Optional[int] = None) -> Tuple[Optional[List[Any]], Optional[List[int]], int, bool]:
    """
    Parses a complete SQLite record: header + body.
    Returns (values, serial_types, total_record_bytes, is_valid).
    """
    if max_len is None:
        max_len = len(data) - offset
    if max_len < 2 or offset >= len(data):
        return None, None, 0, False

    serial_types, header_len, ok = parse_record_header(data, offset, max_header_len=max_len)
    if not ok or serial_types is None:
        return None, None, 0, False

    val_offset = offset + header_len
    values = []
    for st in serial_types:
        sz = serial_type_size(st)
        if sz is None:
            return None, None, 0, False
        if val_offset + sz > offset + max_len or val_offset + sz > len(data):
            return None, None, 0, False
        val = decode_serial_value(st, data[val_offset:val_offset + sz])
        values.append(val)
        val_offset += sz

    total_bytes = val_offset - offset
    return values, serial_types, total_bytes, True


def calculate_entropy(data: bytes) -> float:
    """Calculates the Shannon entropy (in bits) of a byte sequence."""
    if not data:
        return 0.0
    freq = {}
    for b in data:
        freq[b] = freq.get(b, 0) + 1
    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy
