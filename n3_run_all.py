"""Run every N3 step in order and stop at the first error.
N1 must have been run first (N3 builds its databases from N1's cases).
Step 8 needs the folder that holds the third-party corpora (see the guide):
Run:  python n3_run_all.py --source ext_sqlite_forensic
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--source", nargs="+", required=True,
                help="folder(s) with the third-party corpora, passed to n3_06_get_corpora.py")
args = ap.parse_args()
STEPS = [["n3_00_check_env.py"], ["n3_01_build_databases.py"], ["n3_02_carve.py"], ["n3_03_train_ranker.py"],
         ["n3_04_evaluate.py"], ["n3_05_shift.py"], ["n3_06_get_corpora.py", "--source", *args.source],
         ["n3_07_corpora.py"], ["n3_08_export_for_n5.py"], ["n3_09_make_report.py"]]
t_all = time.time()
for s in STEPS:
    print(f"\n######## {' '.join(s)} ########", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, *s]).returncode != 0:
        sys.exit(f"\nSTOPPED: {s[0]} failed. Read the error above, fix it, then re-run from {s[0]}.")
    print(f"######## {s[0]} finished in {time.time() - t0:.0f}s", flush=True)
print(f"\nAll N3 steps finished in {(time.time() - t_all) / 60:.1f} minutes. See results_n3/SUMMARY_N3.md")
