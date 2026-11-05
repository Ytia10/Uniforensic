"""N3 STEP 8 - Put the two third-party test corpora in place and verify every byte.

Get the files ONE of these ways (see the guide), then point this script at them:
  a) git clone https://github.com/SecurityRonin/sqlite-forensic.git ext_sqlite_forensic
     python n3_06_get_corpora.py --source ext_sqlite_forensic
  b) official downloads (Digital Corpora zip for Nemetz, NIST CFReDS SFT-03 folder),
     unzipped anywhere:   python n3_06_get_corpora.py --source <folder> [<folder> ...]
Files are recognised by MD5, so names and folder layout do not matter; anything that
differs by a single byte from the published corpora is ignored.
Output: data_n3/third_party/nemetz/*, data_n3/third_party/cfreds/*
Run:  python n3_06_get_corpora.py --source <folder>
"""
import argparse
import sys
from pathlib import Path

from ufn1.common import ROOT, banner
from ufn3.common import load_config_n3, path_n3
from ufn3.corpora import MANIFEST, collect, verify

ap = argparse.ArgumentParser()
ap.add_argument("--source", nargs="+", default=[], help="folder(s) that contain the corpora")
args = ap.parse_args()
cfg3 = load_config_n3()
dest = path_n3(cfg3, "data") / "third_party"
sources = [Path(s) if Path(s).is_absolute() else ROOT / s for s in args.source]
bad = [str(s) for s in sources if not s.is_dir()]
if bad:
    sys.exit(f"Not a folder: {', '.join(bad)}")
n, missing = collect(sources, dest)
banner(f"Third-party corpora: {n}/{len(MANIFEST)} files in place (MD5-verified)")
for corpus in ("nemetz", "cfreds"):
    have = sum(1 for k in MANIFEST if k.startswith(corpus) and k not in missing)
    print(f"  {corpus:7s} {have:3d}/{sum(k.startswith(corpus) for k in MANIFEST)} files")
problems = verify(dest)
if problems:
    print("  missing or changed:", ", ".join(problems[:8]), "..." if len(problems) > 8 else "")
    sys.exit("Not all corpus files were found. Check --source (see the guide, step 8).")
print("\nAll files present and identical to the published corpora. Next: python n3_07_corpora.py")
