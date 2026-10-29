"""STEP 3 - Train the teammate's N3 classifier exactly as N3 does, and prove it.

1. ufn5.n3_adapter.train_teammate_classifier() repeats N3's Step 1-2 (same 4
   training databases, same injected negatives, same CalibratedSQLiteClassifier)
   after seeding numpy (N3 does not seed it, so each N3 run differs slightly).
2. The teammate's OWN run_n3_pipeline.run_pipeline() is then run in a scratch
   folder with the same numpy seed, and its reported weights / Platt parameters
   must equal ours bit for bit. If they do not, the step stops.

Output: models_n5/n3_classifier.json, results_n5/n3_equivalence.json
Run:  python n5_01_n3_classifier.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ufn1.common import banner, save_json
from ufn5.common import load_config_n5, path_n5
from ufn5.n3_adapter import N3_DIR, classifier_to_json, train_teammate_classifier

cfg5 = load_config_n5()
mdir, rdir, ddir = path_n5(cfg5, "models"), path_n5(cfg5, "results"), path_n5(cfg5, "data")
seed = cfg5["n3_numpy_seed"]

banner("Train the teammate's N3 classifier (N3 Step 1-2, numpy seeded)")
clf, info = train_teammate_classifier(ddir / "n3_train_dbs", seed)
ours = classifier_to_json(clf)
print(f"  {info['n_train_vectors']} training vectors ({info['n_positive']} positive)")
print(f"  Platt A = {clf.platt_a:.4f}, B = {clf.platt_b:.4f};  in-sample ECE = {info['ece_in_sample']:.4f}")
for n, w in zip(clf.FEATURE_NAMES, clf.weights):
    print(f"    {n:<24s} {w:+.4f}")

banner("Equivalence check: the teammate's own run_pipeline() with the same seed")
code = (f"import sys; sys.path.insert(0, {str(N3_DIR)!r}); import numpy as np; "
        f"np.random.seed({seed}); import run_n3_pipeline as r; r.run_pipeline()")
with tempfile.TemporaryDirectory() as tmp:
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit("The teammate's pipeline failed to run - see the output above.")
    rep = json.loads((Path(tmp) / "n3_forensic_results_report.json").read_text())
theirs = rep["calibration"]
same = (theirs["platt_a"] == ours["platt_a"] and theirs["platt_b"] == ours["platt_b"]
        and [theirs["feature_weights"][n] for n in ours["feature_names"]] == ours["weights"])
eq = {"numpy_seed": seed, "identical": same,
      "ours": {"platt_a": ours["platt_a"], "platt_b": ours["platt_b"], "weights": ours["weights"]},
      "teammate_run_pipeline": {"platt_a": theirs["platt_a"], "platt_b": theirs["platt_b"],
                                "weights": [theirs["feature_weights"][n] for n in ours["feature_names"]],
                                "ece_in_sample": theirs["ece"]},
      "teammate_benchmark_summary": rep["benchmark_summary"],
      "teammate_pragma_sensitivity": rep["pragma_sensitivity"]}
save_json(eq, rdir / "n3_equivalence.json")
print(f"  teammate pipeline: Platt A = {theirs['platt_a']:.4f}, B = {theirs['platt_b']:.4f}, "
      f"ECE = {theirs['ece']:.4f}")
n3 = rep["benchmark_summary"]["n3"]
print(f"  teammate benchmark (their own databases): recall {n3['recall']:.3f}, "
      f"precision {n3['precision']:.3f}, live-row FP {n3['fp_live']}")
if not same:
    raise SystemExit("MISMATCH: our copy of N3's training differs from the teammate's code.")
print("  IDENTICAL: the classifier N5 uses is exactly the teammate's N3 classifier.")
save_json({**ours, **info, "numpy_seed": seed}, mdir / "n3_classifier.json")
print("\nDone. Next: python n5_02_build_and_carve.py")
