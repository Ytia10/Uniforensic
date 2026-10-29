"""Run every N2 step in order and stop at the first error.
N1 and N4 must have been run first (N2 reads their data and models).
Run:  python n2_run_all.py
"""
import subprocess
import sys
import time

STEPS = ["n2_00_check_env.py", "n2_01_shadow_pool.py", "n2_02_explanations.py",
         "n2_03_release_attack.py", "n2_04_distinguishability.py", "n2_05_shadow_models.py",
         "n2_06_case_mia.py", "n2_07_make_report.py"]
t_all = time.time()
for s in STEPS:
    print(f"\n######## {s} ########", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"\nSTOPPED: {s} failed. Read the error above, fix it, then re-run from {s}.")
    print(f"######## {s} finished in {time.time() - t0:.0f}s", flush=True)
print(f"\nAll N2 steps finished in {(time.time() - t_all) / 60:.1f} minutes. See results_n2/SUMMARY_N2.md")
