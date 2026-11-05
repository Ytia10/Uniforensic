"""Run every N5b step in order (N5 with the alternative N3 carver; about 30-90 minutes
on a 2-core laptop CPU). Needs the alternative N3 run first (python n3_run_all.py, then
n3_06_get_corpora.py + n3_07_corpora.py). Each step stops the chain if it fails.
Run:  python n5b_run_all.py
"""
import subprocess
import sys
import time

STEPS = ["n5b_00_check_env.py", "n5b_01_carver.py", "n5b_01b_teammate_on_corpora.py",
         "n5b_02_build_and_carve.py",
         "n5b_03_calibration.py", "n5b_04_features.py", "n5b_05_train_variants.py",
         "n5b_06_evaluate.py", "n5b_07_flag.py", "n5b_08_conformal.py", "n5b_09_federated.py",
         "n5b_10_case_export.py", "n5b_11_dp_release.py", "n5b_12_make_report.py"]

t0 = time.time()
for s in STEPS:
    print(f"\n######## {s} ########", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"\nStep {s} failed - fix the message above and re-run from this step.")
    print(f"######## {s} finished in {time.time() - t:.0f}s", flush=True)
print(f"\nAll N5b steps finished in {(time.time() - t0) / 60:.1f} min.")
