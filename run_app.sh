#!/usr/bin/env bash
# UniForensic Workbench - start without Docker (macOS / Linux), from this folder's .venv.
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn ufapp.main:app --port 8000
