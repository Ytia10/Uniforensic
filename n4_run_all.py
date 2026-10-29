"""Run every N4 step in order and stop at the first error.
N1 must have been run first (its data/ and models/mlp.pt are N4's inputs).
Run:  python n4_run_all.py
"""
import subprocess
import sys
import time

STEPS = ["n4_00_check_env.py", "n4_01_case_export.py", "n4_02_partition.py",
         "n4_03_train_baselines.py", "n4_04_train_federated.py", "n4_05_poisoning.py",
         "n4_06_score_models.py", "n4_07_make_report.py"]
t_all = time.time()
for s in STEPS:
    print(f"\n######## {s} ########", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"\nSTOPPED: {s} failed. Read the error above, fix it, then re-run from {s}.")
    print(f"######## {s} finished in {time.time() - t0:.0f}s", flush=True)
print(f"\nAll N4 steps finished in {(time.time() - t_all) / 60:.1f} minutes. See results_n4/SUMMARY_N4.md")
