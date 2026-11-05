"""N5b settings: config_n5b.yaml (a copy of config_n5.yaml with *_n5b folders and a
`carver` block). Every other helper is N5's own (ufn5.common)."""
from __future__ import annotations

import hashlib

import yaml

from ufn1.common import ROOT

UFN3_DIR = ROOT / "ufn3"


def load_config_n5b() -> dict:
    with open(ROOT / "config_n5b.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ufn3_md5() -> dict[str, str]:
    """md5 of every file of the alternative N3 package, recorded in the N5b manifest."""
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(UFN3_DIR.glob("*.py")) + [UFN3_DIR / "corpora_md5.json"] if p.exists()}
