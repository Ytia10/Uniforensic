"""Run every step in order and stop at the first error.
Use this AFTER you have run the steps one by one at least once (so you know
what each prints). Pass --skip-attack to leave out the optional step 12.
Run:  python run_all.py   or   python run_all.py --skip-attack
"""
import subprocess
import sys
import time

STEPS = ["00_check_env.py", "01_generate_benchmark.py", "02_build_features.py",
         "03_sanity_check.py", "04_train_models.py", "05_oracle_check.py",
         "06_run_explainers.py", "07_score_explanations.py", "08_ffidelity.py",
         "09_acceptance_test.py", "10_scaffold_attack.py", "11_make_report.py"]
if "--skip-attack" in sys.argv:
    STEPS.remove("10_scaffold_attack.py")

t_all = time.time()
for s in STEPS:
    print(f"\n######## {s} ########", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"\nSTOPPED: {s} failed. Read the error above, fix it, then re-run from {s}.")
    print(f"######## {s} finished in {time.time() - t0:.0f}s", flush=True)
print(f"\nAll steps finished in {(time.time() - t_all) / 60:.1f} minutes. See results/SUMMARY.md")
