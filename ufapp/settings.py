"""Paths and settings of the workbench. Everything it writes stays under app_data/."""
from __future__ import annotations

from pathlib import Path

import yaml

from ufn1.common import ROOT

with open(ROOT / "config_app.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

APP_DATA = ROOT / CFG["app_data"]
RUNS = APP_DATA / "runs"
UPLOADS = APP_DATA / "uploads"
for p in (APP_DATA, RUNS, UPLOADS):
    p.mkdir(parents=True, exist_ok=True)

WEB_DIST = ROOT / "web" / "dist"          # the built React app (shipped prebuilt)

# Figure folders the app may serve (read-only). N3's figures are the TEAMMATE's,
# produced live by their own pipeline (see n3 page), so figures_n3 is not listed.
FIGURE_DIRS = {"n1": ROOT / "figures", "n2": ROOT / "figures_n2", "n4": ROOT / "figures_n4",
               "n5": ROOT / "figures_n5"}


def run_dir(run_id: str) -> Path:
    p = RUNS / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
