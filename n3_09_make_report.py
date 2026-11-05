"""N3 STEP 11 - Figures and a one-page summary of YOUR N3 numbers.

Output: figures_n3/fig12_calibration.png, fig13_conditions.png, fig14_third_party.png,
        results_n3/SUMMARY_N3.md, results_n3/run_manifest_n3.json
Run:  python n3_09_make_report.py
"""
import importlib
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

import apsw                                                          # noqa: E402
from ufn1.common import ROOT, load_json, save_json, sha256_file      # noqa: E402
from ufn3.common import load_config_n3, path_n3                      # noqa: E402
from ufn3.truth import reliability                                   # noqa: E402

cfg3 = load_config_n3()
data, models, res, fig_dir = (path_n3(cfg3, k) for k in ("data", "models", "results", "figures"))
need = ["test_methods.csv", "test_by_condition.csv", "evidence_recovery.csv", "reliability_test.csv",
        "shift_holdout.csv", "corpora_summary.csv", "corpora_records.csv.gz", "dissect_synthetic.csv"]
missing = [n for n in need if not (res / n).exists()]
if missing:
    raise SystemExit("Missing results: " + ", ".join(missing) + " - run steps 6-9 first.")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"strict": "#52514e", "heuristic": "#eb6834", "ranker_cal": "#2a78d6", "abl_content": "#9a7fd1",
       "abl_no_pay": "#e3a72f", "sqlite_dissect": "#1baf7a"}
LAB = {"strict": "strict carving", "heuristic": "fixed heuristic score", "ranker_cal": "calibrated ranker (N3)",
       "abl_content": "ranker + content feature", "abl_no_pay": "ranker without payment-app tables",
       "sqlite_dissect": "SQLite Dissect 1.0.0"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def tidy(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


mt = pd.read_csv(res / "test_methods.csv").set_index("method")
bc = pd.read_csv(res / "test_by_condition.csv")
ev = pd.read_csv(res / "evidence_recovery.csv")
rel = pd.read_csv(res / "reliability_test.csv")
sh = pd.read_csv(res / "shift_holdout.csv")
cs = pd.read_csv(res / "corpora_summary.csv")
cr = pd.read_csv(res / "corpora_records.csv.gz")
sd = pd.read_csv(res / "dissect_synthetic.csv", keep_default_na=False)
meta = load_json(models / "ranker.json")

# ------------------------------------------------ Fig 12: does "0.8" mean 80 %?
nem = cr[(cr.corpus == "nemetz") & cr.group.isin(cfg3["nemetz"]["categories"]) & (cr.neutral == 0)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
panels = [(axes[0], "Synthetic test databases (N1 test cases)",
           {m: rel[rel.method == m][["mean_predicted", "observed", "count"]].to_numpy() for m in ("heuristic", "ranker_cal")}),
          (axes[1], "Nemetz corpus, record categories (never trained on)",
           {m: np.array(reliability(nem.y, nem[c], 10)) for m, c in (("heuristic", "heuristic"), ("ranker_cal", "p"))})]
for ax, title, curves in panels:
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    for m, arr in curves.items():
        if len(arr):
            ax.plot(arr[:, 0], arr[:, 1], color=COL[m], lw=1.5, alpha=0.6)
            ax.scatter(arr[:, 0], arr[:, 1], s=12 + 18 * np.log10(arr[:, 2] + 1), color=COL[m], zorder=3,
                       label=LAB[m])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence the method gives a recovered record")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    tidy(ax, "both")
axes[0].set_ylabel("share of those records that are really deleted rows")
for ax in axes:
    ax.text(0.98, 0.10, "dot size = records in the bin (log scale)", fontsize=7, color=INK2,
            transform=ax.transAxes, ha="right")
axes[0].text(0.62, 0.55, "perfect calibration", rotation=40, color=INK2, fontsize=8)
e_te = mt.loc["ranker_cal", "ece"], mt.loc["heuristic", "ece"]
r_n = cs[(cs.corpus == "nemetz") & (cs.group == "records (07+0C+0D+0E)")].set_index("method")
axes[0].legend(frameon=False, loc="center left", fontsize=9)
axes[0].text(0.98, 0.04, f"ECE: ranker {e_te[0]:.3f}, heuristic {e_te[1]:.3f}", ha="right", fontsize=8, color=INK2)
axes[1].text(0.98, 0.04, f"ECE: ranker {r_n.loc['ranker_cal', 'ece']:.3f}, heuristic {r_n.loc['heuristic', 'ece']:.3f}",
             ha="right", fontsize=8, color=INK2)
fig.suptitle("The ranker's confidence can be read as a probability; a fixed heuristic score cannot",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig12_calibration.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 13: what survives, what is found, per condition
order = [c for c in cfg3["conditions"] if c in set(bc.condition)]
s_ = bc[bc.method == "strict"].set_index("condition").loc[order]
r_ = bc[bc.method == "ranker_cal"].set_index("condition").loc[order]
fig, ax = plt.subplots(figsize=(10.5, 4.3))
x = np.arange(len(order))
ax.bar(x - 0.27, s_.present / s_.deleted, 0.26, color=GRID, edgecolor=INK2, lw=0.6,
       label="deleted rows whose bytes still exist (ceiling)")
ax.bar(x, s_.tp / s_.deleted, 0.26, color=COL["strict"], label=LAB["strict"])
ax.bar(x + 0.27, r_.tp / r_.deleted, 0.26, color=COL["ranker_cal"], label=LAB["ranker_cal"])
ax.set_xticks(x, [o.replace("_", "\n") for o in order])
ax.set_ylabel("share of ALL deleted rows")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, loc="upper left")
tidy(ax)
fig.suptitle("SQLite settings decide what survives; the ranker recovers almost all of what does",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig13_conditions.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 14: third-party corpora
methods = ["strict", "heuristic", "ranker_cal", "abl_content", "abl_no_pay", "sqlite_dissect"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
for ax, (corpus, group, title) in zip(axes, [("nemetz", "records (07+0C+0D+0E)", "Nemetz corpus, categories 07/0C/0D/0E"),
                                             ("nist_sft03", "all 4 variants", "NIST CFReDS SFT-03 (Android + iOS)")]):
    s = cs[(cs.corpus == corpus) & (cs.group == group)].set_index("method").loc[methods]
    x = np.arange(len(methods))
    ax.bar(x - 0.2, s.precision.fillna(0), 0.38, color=[COL[m] for m in methods], label="precision")
    ax.bar(x + 0.2, s.recall_all.fillna(0), 0.38, color=[COL[m] for m in methods], alpha=0.45, hatch="//",
           edgecolor=SURF, label="recall (of all deleted rows)")
    ceil = s.present.iloc[0] / s.deleted.iloc[0]
    ax.axhline(ceil, color=INK2, lw=1, ls=":")
    ax.text(len(methods) - 0.5, ceil + 0.01, "recall ceiling", ha="right", fontsize=8, color=INK2)
    short = {"strict": "strict\ncarving", "heuristic": "fixed\nheuristic", "ranker_cal": "calibrated\nranker (N3)",
             "abl_content": "ranker +\ncontent", "abl_no_pay": "ranker w/o\npayment app", "sqlite_dissect": "SQLite\nDissect"}
    ax.set_xticks(x, [short[m] for m in methods], fontsize=8)
    if s.loc["sqlite_dissect", "crashed_on"] > 0:
        top = np.nanmax([s.loc["sqlite_dissect", "precision"], s.loc["sqlite_dissect", "recall_all"], 0])
        ax.text(x[-1], top + 0.03, f"crashed on\n{int(s.loc['sqlite_dissect', 'crashed_on'])} DBs", ha="center",
                fontsize=7, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_ylim(0, 1.08)
    tidy(ax)
axes[0].set_ylabel("solid = precision, hatched = recall")
fig.suptitle("Databases N3 never trained on: the calibrated ranker keeps precision high without giving up recall",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig14_third_party.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ summary
stats = pd.read_csv(data / "dbstats_test.csv")
ci = load_json(res / "test_ci_ranker_cal.json")
n_s = cs[(cs.corpus == "nist_sft03") & (cs.group == "all 4 variants")].set_index("method")
sd_rep = sd.tp.sum() + sd.fp.sum() + sd.live_fp.sum()


def fmt(x) -> str:
    return "-" if pd.isna(x) else f"{x:.3f}"


def row(m, src):
    r = src.loc[m]
    return (f"| {LAB.get(m, m)} | {int(r.reported)} | {fmt(r.precision)} | {fmt(r.recall_all)} | "
            f"{fmt(r.recall_present)} | {fmt(r.ece)} |")


lines = [
    "# UniForensic N3 - summary of YOUR run", "",
    f"SQLite engine used to build the databases: {apsw.sqlite_lib_version()} (APSW {apsw.apswversion()}).",
    f"Synthetic test: {len(stats)} databases from {stats.case_id.nunique()} N1 test cases; "
    f"{int(stats.deleted.sum())} deleted rows, {int(stats.present.sum())} still present in the bytes.", "",
    "## Synthetic test databases", "",
    "| method | reported | precision | recall (all deleted) | recall (present) | ECE |", "|---|---|---|---|---|---|",
    *[row(m, mt) for m in ("strict", "heuristic", "ranker_raw", "ranker_cal", "ranker_cal_t95")], "",
    f"Calibrated ranker, 95% intervals (resampling databases): precision {ci['precision'][0]:.3f}-{ci['precision'][1]:.3f}, "
    f"recall (present) {ci['recall_present'][0]:.3f}-{ci['recall_present'][1]:.3f}.",
    "Live rows reported as deleted: 0 for every N3 method (excluded by construction).",
    f"Operating point fixed on the calibration split before testing: p >= 0.5 and p >= {meta['t95']:.3f} (95% precision).", "",
    "## Hold-out tests (step 7): worst case over held-out settings", "",
    f"- unseen SQLite condition: precision >= {sh[sh.holdout == 'condition'].out_precision.min():.3f}, "
    f"recall (present) >= {sh[sh.holdout == 'condition'].out_recall_present.min():.3f}",
    f"- unseen app schema: precision >= {sh[sh.holdout == 'app'].out_precision.min():.3f}, "
    f"recall (present) >= {sh[sh.holdout == 'app'].out_recall_present.min():.3f} "
    f"(worst: {sh[sh.holdout == 'app'].sort_values('out_recall_present').iloc[0].value})", "",
    "## Third-party corpora (never used for training or calibration)", "",
    f"Nemetz record categories: {int(r_n.iloc[0].deleted)} deleted rows, {int(r_n.iloc[0].present)} present.", "",
    "| method | reported | precision | recall (all deleted) | recall (present) | ECE |", "|---|---|---|---|---|---|",
    *[row(m, r_n) for m in methods], "",
    f"NIST SFT-03: {int(n_s.iloc[0].deleted)} deleted rows (4 variants), all present.", "",
    "| method | reported | precision | recall (all deleted) | recall (present) | ECE |", "|---|---|---|---|---|---|",
    *[row(m, n_s) for m in methods], "",
    f"SQLite Dissect crashed on {int(r_n.loc['sqlite_dissect', 'crashed_on'])} of the Nemetz record databases and "
    f"{int(n_s.loc['sqlite_dissect', 'crashed_on'])} of 4 NIST databases (rows written before a crash are counted).", "",
    f"SQLite Dissect on {len(sd)} synthetic test databases: precision {sd.tp.sum() / max(1, sd_rep):.3f}, "
    f"{int(sd.live_fp.sum())} live rows reported as deleted, crashed on {int((sd.error != '').sum())}.", "",
    "## Planted evidence (N1) that was deleted", "",
    "| pattern | deleted | bytes present | recovered (ranker) | recovered (strict) |", "|---|---|---|---|---|",
    *[f"| {r.pattern_id} | {r.deleted} | {r.present:.1%} | {r.recovered:.1%} | {r.recovered_strict:.1%} |"
      for _, r in ev.iterrows()], "",
    "Figures: figures_n3/fig12_calibration.png, fig13_conditions.png, fig14_third_party.png",
]
(res / "SUMMARY_N3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
vers = {m: getattr(importlib.import_module(m), "__version__", "") for m in
        ("numpy", "pandas", "sklearn", "matplotlib", "yaml", "joblib")}
vers["apsw"] = apsw.apswversion()
vers["sqlite_bundled"] = apsw.sqlite_lib_version()
save_json({"python": sys.version.split()[0], "platform": platform.platform(), "versions": vers,
           "config_n3": cfg3, "ranker": meta,
           "sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                      sorted(list(res.glob("*.csv")) + [models / "ranker.json"])}},
          res / "run_manifest_n3.json")
print((res / "SUMMARY_N3.md").read_text(encoding="utf-8"))
print("Done. Figures in figures_n3/, summary in results_n3/SUMMARY_N3.md")
