"""Run every N5 step in order (about 30-60 minutes on a 2-core laptop CPU).
Each step stops the chain if it fails.  Run:  python n5_run_all.py
"""
import subprocess
import sys
import time

STEPS = ["n5_00_check_env.py", "n5_01_n3_classifier.py", "n5_02_build_and_carve.py",
         "n5_03_calibration.py", "n5_04_features.py", "n5_05_train_variants.py",
         "n5_06_evaluate.py", "n5_07_flag.py", "n5_08_conformal.py", "n5_09_federated.py",
         "n5_10_case_export.py", "n5_11_dp_release.py", "n5_12_make_report.py"]

t0 = time.time()
for s in STEPS:
    print(f"\n######## {s} ########", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"\nStep {s} failed - fix the message above and re-run from this step.")
    print(f"######## {s} finished in {time.time() - t:.0f}s", flush=True)
print(f"\nAll N5 steps finished in {(time.time() - t0) / 60:.1f} min.")
