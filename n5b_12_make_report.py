"""N5b STEP 14 - Figures, example case reports and a one-page summary of the N5b numbers
(N5 re-run with the alternative N3 carver), plus a side-by-side table with N5 when
results_n5/ is present (read-only).

Output: figures_n5b/fig15_recovery.png, fig16_calibration.png, fig17_triage_vs_deletion.png,
        fig18_false_incrimination.png, fig19_flag.png, fig20_conformal.png,
        results_n5b/example_reports.md, results_n5b/SUMMARY_N5B.md, results_n5b/run_manifest_n5b.json
Run:  python n5b_12_make_report.py
"""
import importlib
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

from ufn1.common import ROOT, load_json, save_json, sha256_file     # noqa: E402
from ufn1.schema import CLASSES, DECISIVE_FEATURES                  # noqa: E402
from ufn5.common import path_n5                                     # noqa: E402
from ufn5.devicedb import TABLES                                    # noqa: E402
from ufn5b.common import load_config_n5b, ufn3_md5                  # noqa: E402
from ufn5.sets import BENIGN, load                                  # noqa: E402

cfg5 = load_config_n5b()
ddir, mdir, res, fig = (path_n5(cfg5, k) for k in ("data", "models", "results", "figures"))
levels = cfg5["scenario"]["levels"]
LV = [f"{int(x * 100)}%" for x in levels]
COL = {"L": "#6250d6", "A": "#eb6834", "Af": "#d9a400", "B": "#1baf7a", "C": "#2a78d6", "N1": "#52514e"}
NAME = {"L": "L  live only", "A": "A  merge (N3 p >= 0.5)", "Af": "Af merge, teammate filter",
        "B": "B  confidence-weighted", "C": "C  two-channel"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


rec_info = load_json(res / "n3_on_n1_databases.json")
grid = pd.read_csv(res / "eval_grid.csv", dtype={"rho": str, "level": str, "foreign_rate": str})
expl = pd.read_csv(res / "eval_explain.csv", dtype={"rho": str})
flag = pd.read_csv(res / "flag.csv", dtype={"rho": str, "level": str})
agree = pd.read_csv(res / "flag_attribution_agreement.csv", dtype={"rho": str})
conf = pd.read_csv(res / "conformal.csv", dtype={"rho": str, "level": str})
fed = pd.read_csv(res / "federated.csv", dtype={"level": str})
dp = pd.read_csv(res / "dp_release.csv")
case = load_json(res / "case_export_check.json")
carv = load_json(res / "carver_ufn3.json")
same = load_json(res / "same_truth_as_n5.json")


def by_level(df, variant, rho, metric, rate="0.0"):
    d = df[(df.variant == variant) & (df.rho == rho) & (df.foreign_rate == rate)
           & ~df.level.isin(["benign_all"])].copy()
    d["lv"] = d.level.astype(float)
    g = d.groupby("lv")[metric]
    return g.mean().reindex(levels).to_numpy(), g.std().reindex(levels).fillna(0).to_numpy()


# ---------------------------------------------------------- Fig 15: recovery
dt = pd.read_csv(ddir / "test_deleted.csv.gz", keep_default_na=False, dtype=str)
rt = pd.read_csv(ddir / "test_recovered.csv.gz", keep_default_na=False, dtype=str)
got = set(zip(rt.version_id, rt.match_id))
dt["recovered"] = [(v, a) in got for v, a in zip(dt.version_id, dt.artifact_id)]
ev = dt[(dt.role == "evidence")]
fig_, ax = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(TABLES))
for i, (b, c) in enumerate([("immediate", "#eb6834"), ("clean-up", "#2a78d6")]):
    r = ev[ev.behaviour == b].groupby("table").recovered.mean().reindex(TABLES)
    ax[0].bar(x + (i - 0.5) * 0.38, r, 0.38, color=c, label=f"deleted {b}")
ax[0].set_xticks(x, TABLES)
ax[0].set_ylim(0, 1)
ax[0].set_ylabel("share of deleted evidence recovered")
ax[0].set_title("(a) deleted evidence recovered by the alternative N3,\nby when the suspect deleted it")
ax[0].legend(frameon=False, fontsize=8)
fz = pd.Series(rec_info["fuzzy_filter_discard_rate_by_table"]).reindex(TABLES)
ax[1].bar(x, fz, 0.55, color="#d9a400")
for xi, v in zip(x, fz):
    ax[1].text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
ax[1].set_xticks(x, TABLES)
ax[1].set_ylim(0, 1.1)
ax[1].set_ylabel("correctly recovered deleted records")
ax[1].set_title("(b) share the teammate's fuzzy live-row\nfilter would discard (they are NOT live)")
for a in ax:
    tidy(a)
fig_.tight_layout()
fig_.savefig(fig / "fig15_recovery.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- Fig 16: calibration
rel = pd.read_csv(res / "reliability.csv")
cal = rec_info["calibration"]
fig_, ax = plt.subplots(figsize=(5, 4.2))
ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", label="perfect calibration")
for name, c, lab in (("raw", "#eb6834", f"N3 p as shipped (ECE {cal['ece_raw']:.3f})"),
                     ("recalibrated", "#2a78d6", f"after isotonic (ECE {cal['ece_recalibrated']:.3f})")):
    d = rel[rel.p == name]
    ax.scatter(d.mean_p, d.share_genuine, s=np.clip(d["count"] / d["count"].max() * 300, 12, 300),
               color=c, alpha=0.8, label=lab)
ax.set_xlabel("confidence given to the recovered record")
ax.set_ylabel("share that really existed")
ax.set_title("Alternative N3 confidence on N1 databases\n(test split; dot size = number of records)")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8, loc="lower right")
tidy(ax)
fig_.tight_layout()
fig_.savefig(fig / "fig16_calibration.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- Fig 17: triage vs deletion
fig_, ax = plt.subplots(1, 2, figsize=(11, 4))
for v in ("L", "A", "B", "C"):
    m, s = by_level(grid, v, "0.0", "macro_f1")
    ax[0].errorbar(range(len(levels)), m, yerr=s, marker="o", ms=4, capsize=3, color=COL[v], label=NAME[v])
m, _ = by_level(grid, "A", "N1", "macro_f1")
ax[0].plot(range(len(levels)), m, ls="--", color=COL["N1"], label="N1 model unchanged, view A")
e = expl[expl.rho == "0.0"]
for v in ("L", "A", "B", "C"):
    d = e[e.variant == v].groupby("level").ter
    ax[1].errorbar(range(len(levels)), d.mean().reindex(levels), yerr=d.std().reindex(levels),
                   marker="o", ms=4, capsize=3, color=COL[v], label=NAME[v])
ax[0].set_ylabel("macro-F1 (test, clean benign devices)")
ax[1].set_ylabel("TER: right class AND IG finds >= 2/3 of\nthe still-visible planted evidence")
for a, t in zip(ax, ("(a) triage accuracy", "(b) right decision for the right reason")):
    a.set_xticks(range(len(levels)), LV)
    a.set_xlabel("share of the planted evidence the suspect deleted")
    a.set_ylim(0, 1)
    a.set_title(t)
    tidy(a)
ax[0].legend(frameon=False, fontsize=8, loc="lower left")
fig_.tight_layout()
fig_.savefig(fig / "fig17_triage_vs_deletion.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- Fig 18: false incrimination
b = grid[grid.level == "benign_all"]
fig_, ax = plt.subplots(figsize=(9, 4))
items = [(v, r) for v in ("L", "A", "B", "C") for r in ("0.0", "0.25")] + [("A", "N1")]
x = np.arange(len(items))
for i, (fr, c) in enumerate([("clean", "#9fc3ec"), ("residue", "#eb6834")]):
    vals = [b[(b.variant == v) & (b.rho == r) & (b.foreign_rate == fr)].false_incrimination.mean() for v, r in items]
    ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=c,
           label="clean benign device" if fr == "clean" else "benign device with foreign residue")
lab = [f"{v}\nrho={r}" if r != "N1" else "A\nN1 model" for v, r in items]
ax.set_xticks(x, lab, fontsize=8)
ax.set_ylabel("share of innocent devices triaged as an offence")
ax.set_title("Foreign residue: correctly carved records that are not the suspect's")
ax.legend(frameon=False, fontsize=8)
tidy(ax)
fig_.tight_layout()
fig_.savefig(fig / "fig18_false_incrimination.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- Fig 19: flag
fig_, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = flag[(flag.group == "offence_correct") & (flag.rho == "0.0")].copy()
w["lv"] = w.level.astype(float)
for v in ("A", "B", "C"):
    d = w[w.variant == v].groupby("lv").flag_rate
    ax[0].errorbar(range(len(levels)), d.mean().reindex(levels), yerr=d.std().reindex(levels), marker="o",
                   ms=4, capsize=3, color=COL[v], label=NAME[v])
ax[0].set_xticks(range(len(levels)), LV)
ax[0].set_xlabel("share of the planted evidence the suspect deleted")
ax[0].set_ylabel("correct offence decisions flagged")
ax[0].set_title("(a) workload: decisions that rest on recovered records")
ax[0].legend(frameon=False, fontsize=8)
fc = flag[flag.group == "false_incrimination_residue_caused_by_recovered"]
vs = [("C", "0.0"), ("C", "0.25")]
x = np.arange(len(vs))
ax[1].bar(x - 0.2, [1.0, 1.0], 0.4, color="#eb6834", label="counterfactual flag (exact test)")
ax[1].bar(x + 0.2, [agree[agree.rho == r].attr_catch_rate_recovery_caused_fi.mean() for _, r in vs], 0.4,
          color="#2a78d6", label="attribution flag (IG share >= 50%)")
ax[1].set_xticks(x, [f"C rho={r}\n({fc[(fc.variant == v) & (fc.rho == r)].share_of_fi.mean():.0%} of residue "
                     f"false incriminations\nare caused by recovered records)" for v, r in vs], fontsize=8)
ax[1].set_ylim(0, 1.3)
ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[1].set_ylabel("share of those caused cases flagged")
ax[1].set_title("(b) innocent residue devices wrongly triaged\nbecause of recovered records: flagged?")
ax[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
for a in ax:
    tidy(a)
fig_.tight_layout()
fig_.savefig(fig / "fig19_flag.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- Fig 20: conformal
fig_, ax = plt.subplots(1, 2, figsize=(11, 3.8))
rates = cfg5["foreign_rates_test"]
for v, r in (("A", "0.0"), ("C", "0.0"), ("C", "0.25")):
    d = conf[(conf.variant == v) & (conf.rho == r) & ~conf.level.isin(["mixture", "benign_residue"])]
    cf = d.groupby("foreign_rate").coverage_full.mean().reindex(rates)
    cl = d.groupby("foreign_rate").coverage_live.mean().reindex(rates)
    ls = "-" if r == "0.0" else ":"
    ax[0].plot(rates, cf, marker="o", ms=4, ls=ls, color=COL[v], label=f"{v} rho={r}: full set")
    ax[0].plot(rates, cl, marker="s", ms=4, ls=ls, color=COL[v], alpha=0.45, label=f"{v} rho={r}: live set")
ax[0].axhline(1 - cfg5["conformal"]["alpha"], color=INK2, lw=0.8, ls="--")
ax[0].set_xlabel("share of benign test devices with foreign residue")
ax[0].set_ylabel("coverage (all levels pooled)")
ax[0].set_title("(a) coverage of the full set and of the live set")
ax[0].legend(frameon=False, fontsize=7)
mx = conf[conf.level == "mixture"]
vs = ["L", "A", "B", "C"]
x = np.arange(len(vs))
ax[1].bar(x - 0.2, [mx[(mx.variant == v) & (mx.rho == "0.0")].size_full.mean() for v in vs], 0.4,
          color="#2a78d6", label="full set (with recovered records)")
ax[1].bar(x + 0.2, [mx[(mx.variant == v) & (mx.rho == "0.0")].size_live.mean() for v in vs], 0.4,
          color="#9fc3ec", label="live set (recovered records removed)")
ax[1].set_xticks(x, vs)
ax[1].set_ylabel("mean classes per 90% set")
ax[1].set_title("(b) how much recovered records narrow the conclusion")
ax[1].legend(frameon=False, fontsize=8)
for a in ax:
    tidy(a)
fig_.tight_layout()
fig_.savefig(fig / "fig20_conformal.png", dpi=200)
plt.close(fig_)

# ---------------------------------------------------------- example case reports
TE = load(ddir, "test")
p = np.load(res / "pred" / "C_rho0.25_s2026.npz")
row = conf[(conf.variant == "C") & (conf.rho == "0.25") & (conf.seed == 2026)].iloc[0]
P, P0 = p["test_proba"], p["test_proba0"]
S, S0 = (1 - P) <= row.q_full, (1 - P0) <= row.q_live
pred, pred0 = P.argmax(1), P0.argmax(1)
recn = pd.read_csv(ddir / "test_recovered.csv.gz", keep_default_na=False, dtype=str)
recn["p_raw"] = recn.p_raw.astype(float)
recal = load_json(mdir / "recalibration.json")
from ufn5.calib import apply_isotonic, to_percent                   # noqa: E402
recn["conf"] = 0.0
for t in TABLES:
    m = recn.table == t
    recn.loc[m, "conf"] = to_percent(apply_isotonic(recal[t], recn.loc[m, "p_raw"].to_numpy()))
raw_C = pd.read_csv(ddir / "test_raw_C.csv.gz", index_col=0)
vid = np.array([k["version_id"] for k in TE["keys"]])
y = TE["y"]
pick = {
    "Offence, all planted evidence deleted":
        np.where((y != BENIGN) & np.isclose(TE["level"], 1.0) & (pred == y) & (pred != pred0))[0],
    "Innocent device holding foreign residue":
        np.where((y == BENIGN) & TE["foreign"].astype(bool) & (pred != BENIGN) & (pred0 == BENIGN))[0],
    "Offence, nothing of the evidence deleted":
        np.where((y != BENIGN) & np.isclose(TE["level"], 0.0) & (pred == y) & (pred == pred0))[0]}
lines = ["# Example N5b case reports (alternative N3; variant C, rho = 0.25, seed 2026, test split)", "",
         "Generated automatically from the run. 'Truth' lines are what the planted answer key says;",
         "an examiner's report would not have them.", ""]
for title, idx in pick.items():
    if not len(idx):
        continue
    i = idx[0]
    r = recn[recn.version_id == vid[i]]
    rc = raw_C.loc[vid[i], [f"rec_{f}" for f in DECISIVE_FEATURES]]
    feats = ", ".join(f"{f.removeprefix('rec_')} +{v:.1f}" for f, v in rc[rc.abs() > 1e-9].items()) or "none"
    lines += [f"## {title}", "", f"Case version `{vid[i]}`", "",
              f"- Triage: **{CLASSES[pred[i]]}** (probability {P[i, pred[i]]:.2f}).",
              f"- 90% prediction set with recovered records: {{{', '.join(CLASSES[c] for c in np.where(S[i])[0])}}}.",
              f"- 90% prediction set from live data only: {{{', '.join(CLASSES[c] for c in np.where(S0[i])[0])}}}.",
              f"- Without the recovered records the model says: {CLASSES[pred0[i]]}.",
              f"- Recovered records: {len(r)} (carved from deleted space; calibrated confidence "
              f"{r.conf.min():.2f}-{r.conf.max():.2f}); recovered channel of the decisive features: {feats}.",
              f"- Share of the decisive-feature attribution on recovered records: {p['test_rec_share'][i]:.0%}."]
    if pred[i] != pred0[i]:
        lines.append("- **This conclusion depends on recovered records.** Their confidence says the "
                     "records are intact, not who created them: confirm attribution (who used the "
                     "device, when) before relying on them.")
    else:
        lines.append("- The conclusion does not depend on recovered records: live data alone gives the same class.")
    truth = TE["keys"][i]
    lines += [f"- Truth: {truth['label']}; recovered records by origin: "
              + ", ".join(f"{k} {v}" for k, v in r.label.value_counts().items()), ""]
(res / "example_reports.md").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------- summary
def mean_level(v, rho, lv, metric="macro_f1"):
    d = grid[(grid.variant == v) & (grid.rho == rho) & (grid.foreign_rate == "0.0")
             & (grid.level == str(float(lv)))]
    return d[metric].mean(), d[metric].std()


def bn(v, rho, fr):
    return b[(b.variant == v) & (b.rho == rho) & (b.foreign_rate == fr)].false_incrimination.mean()


pub_lines = []
if (res / "carvers_on_public_corpora.csv").exists():
    pc = pd.read_csv(res / "carvers_on_public_corpora.csv")
    pc = pc[pc.group.isin(["records (07+0C+0D+0E)", "all 4 variants"])]
    pub_lines = ["", "### Both carvers on the same public corpora (n5b_01b; p >= 0.5)", "",
                 "| corpus | method | deleted | present | reported | correct | precision | recall (all) | "
                 "recall (present) | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pc.iterrows():
        pub_lines.append(f"| {r.corpus} | {r.method} | {int(r.deleted)} | {int(r.present)} | {int(r.reported)} | "
                         f"{int(r.tp)} | {r.precision:.3f} | {r.recall_all:.3f} | {r.recall_present:.3f} | "
                         f"{r.ece:.3f} |")
    pub_lines.append("")
pubtxt = "; ".join(f"{r['corpus']} ({r['group']}): precision {r['precision']:.3f}, recall of present records "
                   f"{r['recall_present']:.3f}, ECE {r['ece']:.3f}" for r in carv["public_corpus_results"])
S_ = ["# UniForensic N5b - summary of this run (N5 with the alternative N3 carver)", "",
      "## Alternative N3 (ufn3) on N1's databases", "",
      f"- Carver: {carv['carver']}; ranker {carv['ranker_file']} (sha256 {carv['ranker_sha256'][:16]}...), "
      f"{carv['ranker_meta']['calibration']} calibration, report threshold p >= {carv['ranker_meta']['report_threshold']}.",
      f"- Same carver on public corpora (n3_07, not used for training): {pubtxt}.",
      *pub_lines,
      f"- Same planted + deleted records as N5: " + (f"{same['identical']} ({same['n5b_rows']} test rows)"
                                                     if same["checked"] else "not checked (no data_n5/)") + ".",
      f"- Deleted planted evidence recovered: {rec_info['recovery_rate_by_role']['evidence']:.1%} "
      f"(background {rec_info['recovery_rate_by_role']['background']:.1%}, foreign residue "
      f"{rec_info['recovery_rate_foreign']:.1%}).",
      f"- Recovered records that are fabrications ('wrong'): {rec_info['wrong_share']:.2%}.",
      f"- ECE of N3's p: on N1 test databases "
      f"{cal['ece_raw']:.4f}; after isotonic recalibration {cal['ece_recalibrated']:.4f}.",
      f"- Correctly recovered deleted records the fuzzy live filter would discard: "
      f"{rec_info['fuzzy_filter_discard_rate_all']:.1%} (planted evidence: "
      f"{rec_info['fuzzy_filter_discard_rate_by_role']['evidence']:.1%}).", "",
      "## Triage (macro-F1, mean +- SD over 3 seeds, rho = 0, clean benign devices)", "",
      "| variant | " + " | ".join(f"deleted {x}" for x in LV) + " |", "|---|" + "---|" * len(LV)]
for v in ("L", "A", "Af", "B", "C"):
    S_.append(f"| {v} | " + " | ".join("{:.3f} +- {:.3f}".format(*mean_level(v, "0.0", lv)) for lv in levels) + " |")
S_.append("| A, N1 model unchanged | " + " | ".join(f"{mean_level('A', 'N1', lv)[0]:.3f}" for lv in levels) + " |")
S_ += ["", "## Explanations (TER, rho = 0)", "",
       "| variant | " + " | ".join(LV) + " |", "|---|" + "---|" * len(LV)]
for v in ("L", "A", "B", "C"):
    d = expl[(expl.variant == v) & (expl.rho == "0.0")].groupby("level").ter.mean().reindex(levels)
    S_.append(f"| {v} | " + " | ".join(f"{x:.3f}" for x in d) + " |")
S_ += ["", "## Innocent devices triaged as offences", "",
       "| variant | rho | clean | with foreign residue |", "|---|---|---|---|"]
for v, r in items:
    S_.append(f"| {v} | {r} | {bn(v, r, 'clean'):.3f} | {bn(v, r, 'residue'):.3f} |")
fcC = fc[(fc.variant == "C") & (fc.rho == "0.0")]
S_ += ["", "## Recovery-dependence flag", "",
       f"- Residue false incriminations caused by recovered records (C, rho 0): "
       f"{fcC.share_of_fi.mean():.1%}; all carry the counterfactual flag (exact test).",
       f"- Attribution flag (C) vs counterfactual: agreement {agree.agreement.mean():.3f}, catches "
       f"{agree.attr_catch_rate_recovery_caused_fi.mean():.1%} of recovery-caused false incriminations.",
       "- Workload (correct offence decisions flagged, C rho 0): " + ", ".join(
           f"{lv}: {w[(w.variant == 'C') & np.isclose(w.lv, x)].flag_rate.mean():.3f}" for lv, x in zip(LV, levels)),
       "", "## Conformal sets (alpha = 0.10)", ""]
for v, r in (("A", "0.0"), ("C", "0.0"), ("C", "0.25")):
    m = mx[(mx.variant == v) & (mx.rho == r)]
    br = conf[(conf.variant == v) & (conf.rho == r) & (conf.level == "benign_residue")]
    S_.append(f"- {v} rho={r}: mixture coverage full {m.coverage_full.mean():.3f} (size {m.size_full.mean():.2f}), "
              f"live {m.coverage_live.mean():.3f} (size {m.size_live.mean():.2f}); on residue devices full "
              f"{br.coverage_full.mean():.3f}, live {br.coverage_live.mean():.3f}.")
S_.append(f"- Own LAC sets identical to MAPIE for "
          f"{load_json(res / 'conformal_mapie_check.json')['min_identical_share']:.1%} of cases (worst model).")
f = fed[fed.level != "benign_residue"].groupby(["variant", "training"]).macro_f1.mean().unstack()
S_ += ["", "## Alignment with N4 and N2", "",
       "- FedAvg on N4's partition (alpha 0.5), mean macro-F1 over levels: " + "; ".join(
           f"{v} federated {f.loc[v, 'federated']:.3f} vs centralised {f.loc[v, 'centralised']:.3f}" for v in f.index),
       f"- CASE/UCO export with RecoveredObjectFacet + ConfidenceFacet: {sum(c['conforms'] for c in case)}/"
       f"{len(case)} files valid (CASE 1.5.0); features from CASE equal step 6 "
       f"(max diff {max(c['max_diff_B'] for c in case):.1e}).",
       "- N2 DP release of recovered reliance, mean abs error by epsilon: " + ", ".join(
           f"{e:g}: {v:.4f}" for e, v in dp.groupby("eps").abs_error.mean().items()), ""]
# ---------------------------------------------------------- side by side with N5 (read-only)
r5 = ROOT / "results_n5"
if (r5 / "eval_grid.csv").exists() and (r5 / "n3_on_n1_databases.json").exists():
    g5 = pd.read_csv(r5 / "eval_grid.csv", dtype={"rho": str, "level": str, "foreign_rate": str})
    i5 = load_json(r5 / "n3_on_n1_databases.json")
    b5 = g5[g5.level == "benign_all"]

    def f1(g, v, lv):
        return g[(g.variant == v) & (g.rho == "0.0") & (g.foreign_rate == "0.0")
                 & (g.level == str(float(lv)))].macro_f1.mean()

    def fi(bb, v, r, fr):
        return bb[(bb.variant == v) & (bb.rho == r) & (bb.foreign_rate == fr)].false_incrimination.mean()

    S_ += ["## Side by side: N5 (teammate's N3) vs N5b (alternative N3), same phones", "",
           "| measure | N5 | N5b |", "|---|---|---|",
           f"| deleted planted evidence recovered | {i5['recovery_rate_by_role']['evidence']:.3f} | "
           f"{rec_info['recovery_rate_by_role']['evidence']:.3f} |",
           f"| deleted background recovered | {i5['recovery_rate_by_role']['background']:.3f} | "
           f"{rec_info['recovery_rate_by_role']['background']:.3f} |",
           f"| foreign residue recovered | {i5['recovery_rate_foreign']:.3f} | {rec_info['recovery_rate_foreign']:.3f} |",
           f"| recovered records that are 'wrong' | {i5['wrong_share']:.4f} | {rec_info['wrong_share']:.4f} |",
           f"| ECE of p, raw (test) | {i5['calibration']['ece_raw']:.4f} | {cal['ece_raw']:.4f} |",
           f"| ECE of p, after isotonic (test) | {i5['calibration']['ece_recalibrated']:.4f} | "
           f"{cal['ece_recalibrated']:.4f} |"]
    for v in ("L", "A", "Af", "B", "C"):
        for lv, x in zip(LV, levels):
            S_.append(f"| macro-F1 {v}, deleted {lv} (rho 0, mean of seeds) | {f1(g5, v, x):.3f} | {f1(grid, v, x):.3f} |")
    for v, r in items:
        S_.append(f"| innocent residue devices triaged as offence, {v} rho {r} | {fi(b5, v, r, 'residue'):.3f} | "
                  f"{bn(v, r, 'residue'):.3f} |")
    S_.append("")
(res / "SUMMARY_N5B.md").write_text("\n".join(S_), encoding="utf-8")
print("\n".join(S_))

# ---------------------------------------------------------- manifest
pk = {}
for mod in ["numpy", "pandas", "scipy", "sklearn", "torch", "captum", "flwr", "case_utils", "mapie", "matplotlib"]:
    try:
        pk[mod] = importlib.import_module(mod).__version__
    except Exception:                                                # noqa: BLE001
        pk[mod] = "?"
outs = sorted([*res.glob("*.csv"), *res.glob("*.json"), *mdir.glob("*.pt"), *mdir.glob("*.json"),
               *ddir.glob("*_versions.csv")])
import sqlite3                                                       # noqa: E402
save_json({"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk,
           "sqlite_library": sqlite3.sqlite_version,
           "carver": "ufn3 (alternative N3)", "ufn3_md5": ufn3_md5(),
           "ranker_sha256": sha256_file(ROOT / cfg5["carver"]["ranker"]),
           "files": {str(o.relative_to(ROOT)): sha256_file(o) for o in outs if o.name != "run_manifest_n5b.json"}},
          res / "run_manifest_n5b.json")
print(f"\nFigures in {fig.relative_to(ROOT)}/, summary in {res.relative_to(ROOT)}/SUMMARY_N5B.md")
