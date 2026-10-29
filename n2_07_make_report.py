"""N2 STEP 9 - Figures and a one-page summary of YOUR N2 numbers.

Output: figures_n2/fig9_privacy_vs_faithfulness.png, fig10_group_size.png,
        fig11_case_mia.png, results_n2/SUMMARY_N2.md, results_n2/run_manifest_n2.json
Run:  python n2_07_make_report.py
"""
import importlib
import math
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from ufn1.common import ROOT, load_json, save_json, sha256_file      # noqa: E402
from ufn2.common import load_config_n2, path_n2                      # noqa: E402
from ufn2.dp import sigma_for                                        # noqa: E402
from ufn2.summaries import SENSITIVITY                               # noqa: E402

cfg2 = load_config_n2()
res, fig_dir = path_n2(cfg2, "results"), path_n2(cfg2, "figures")
eps_list = cfg2["summary"]["epsilons"]
ELAB = [("no noise" if math.isinf(e) else f"{e:g}") for e in eps_list]
SCOL = {"none": "#52514e", "dp": "#eb6834", "secagg": "#2a78d6", "secagg_dp": "#1baf7a"}
SLAB = {"none": "no protection", "dp": "DP noise per agency", "secagg": "secure aggregation",
        "secagg_dp": "secure aggregation + DP"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------ Step 5 table (real deployment)
ra = pd.read_csv(res / "release_attack.csv")
per_seed = ra.groupby(["scheme", "eps", "seed"])[["auc", "tpr_at_1pct_fpr", "evidence_recall3"]].mean()
rel = per_seed.groupby(["scheme", "eps"]).agg(["mean", "std"])


def rel_val(scheme, eps, col):
    e = eps if scheme in ("dp", "secagg_dp") else math.inf
    return rel.loc[(scheme, e), (col, "mean")], rel.loc[(scheme, e), (col, "std")]


# ------------------------------------------------ Fig 9: privacy vs faithfulness
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
x = np.arange(len(eps_list))
for ax, col, title in [(axes[0], "auc", "Membership attack on shared summaries (AUC)"),
                       (axes[1], "evidence_recall3", "Summaries still point at the planted evidence")]:
    for k, scheme in enumerate(cfg2["summary"]["schemes"]):
        mu = [rel_val(scheme, e, col)[0] for e in eps_list]
        sd = [rel_val(scheme, e, col)[1] for e in eps_list]
        dashed = scheme in ("none", "secagg")
        ax.errorbar(x + (k - 1.5) * 0.05, mu, yerr=sd, color=SCOL[scheme], lw=2, marker="o", ms=4,
                    capsize=3, ls="--" if dashed else "-", label=SLAB[scheme])
    ax.set_xticks(x, ELAB)
    ax.set_xlabel("privacy level epsilon (smaller = more private)")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    tidy(ax)
axes[0].axhline(0.5, color=INK2, lw=1, ls=":")
axes[0].text(len(eps_list) - 1, 0.497, "chance", fontsize=8, color=INK2, va="top", ha="center")
axes[0].set_ylim(0.45, 0.65)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("evidence recall@3 of class summaries")
axes[0].set_ylabel("AUC (0.5 = attacker guesses)")
axes[1].legend(frameon=False, loc="lower right", fontsize=9)
fig.suptitle("Secure aggregation + DP removes the leak and keeps the evidence; per-agency DP does not",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig9_privacy_vs_faithfulness.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 10: group size (game)
gm = pd.read_csv(res / "game.csv")
show_eps = [1.0, 8.0, math.inf]
ECOL = {1.0: "#1baf7a", 8.0: "#eb6834", math.inf: "#52514e"}
attacks = [("tracing", "tracing (no training)"), ("logreg_adaptive", "logistic regression"),
           ("rf_adaptive", "random forest")]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
ms = sorted(gm.m.unique())
for ax, (att, title) in zip(axes, attacks):
    for e in show_eps:
        g = gm[(gm.attack == att) & (gm.eps == e)].groupby("m").auc
        mu, n, sd = g.mean().reindex(ms), g.count().reindex(ms), g.std().reindex(ms)
        ci = 1.96 * sd / np.sqrt(n)
        ax.errorbar(range(len(ms)), mu, yerr=ci, color=ECOL[e], lw=2, marker="o", ms=4, capsize=3,
                    label="no noise" if math.isinf(e) else f"epsilon {e:g}")
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(ms)), [str(m) for m in ms])
    ax.set_xlabel("cases in the summary (group size)")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    tidy(ax)
axes[0].set_ylabel("attack AUC (mean, 95% CI)")
axes[0].set_ylim(0.4, 0.95)
axes[2].legend(frameon=False, loc="upper right", fontsize=9)
fig.suptitle("Small summaries leak; epsilon = 1 closes the leak at every group size, epsilon = 8 only reduces it",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig10_group_size.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 11: per-case attacks
cm = pd.read_csv(res / "case_mia.csv")
TORDER = ["overfit_control", "centralised", "federated", "local_only"]
TLAB = {"overfit_control": "overfit control\n(positive control)", "centralised": "centralised",
        "federated": "federated (FedAvg)", "local_only": "local-only\n(8 agencies)"}
STAT = [("variance", "#2a78d6"), ("l1", "#eb6834"), ("l2", "#1baf7a")]
lrt = cm[cm.test == "lrt"].set_index(["target", "statistic"])
fig, ax = plt.subplots(figsize=(8.4, 4))
for k, (st, col) in enumerate(STAT):
    mu = [lrt.loc[(t, st), "auc"] for t in TORDER]
    err = np.array([[lrt.loc[(t, st), "auc"] - lrt.loc[(t, st), "auc_lo"] for t in TORDER],
                    [lrt.loc[(t, st), "auc_hi"] - lrt.loc[(t, st), "auc"] for t in TORDER]])
    ax.bar(np.arange(4) + (k - 1) * 0.26, mu, 0.24, yerr=err, color=col, capsize=3,
           label={"variance": "variance", "l1": "L1 norm", "l2": "L2 norm"}[st],
           error_kw={"ecolor": INK2, "elinewidth": 1})
ax.axhline(0.5, color=INK2, lw=1, ls=":")
ax.set_xticks(range(4), [TLAB[t] for t in TORDER])
ax.set_ylim(0.4, 0.7)
ax.set_ylabel("attack AUC (95% CI)")
ax.legend(frameon=False, loc="upper right", fontsize=9, title="statistic", title_fontsize=9)
ax.set_title("Per-case explanations: only the memorising control leaks", loc="left", color=INK)
tidy(ax)
fig.tight_layout()
fig.savefig(fig_dir / "fig11_case_mia.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ SUMMARY_N2.md
f3 = lambda m, s: f"{m:.3f}" if (s is None or np.isnan(s)) else f"{m:.3f} +/- {s:.3f}"
d = cfg2["summary"]["delta"]
L = ["# N2 results summary (all numbers from this run)\n",
     f"Deployed model: N4 {cfg2['deployed']['method']}, alpha {cfg2['deployed']['alpha']}. "
     f"Explainer: {cfg2['explainer']}. delta = {d:g}. Sensitivity = sqrt(2).\n",
     "## Noise used for each epsilon (analytic Gaussian mechanism)\n",
     "| epsilon | sigma (in units of the clipping bound C) |", "|---|---|"]
for e in eps_list:
    L.append(f"| {'no noise' if math.isinf(e) else f'{e:g}'} | {sigma_for(e, d, SENSITIVITY):.3f} |")
L += [f"\nClipping bound C per seed: " + ", ".join(f"{s}: {c:.3f}" for s, c in ra.groupby('seed').C.first().items()),
      "\n## Step 5 - real deployment: leak and cost per release scheme (mean +/- SD over seeds)\n",
      "| scheme | epsilon | attack AUC | TPR at 1% FPR | evidence recall@3 |", "|---|---|---|---|---|"]
for scheme in cfg2["summary"]["schemes"]:
    for e in (eps_list if scheme in ("dp", "secagg_dp") else [math.inf]):
        L.append(f"| {SLAB[scheme]} | {'no noise' if math.isinf(e) else f'{e:g}'} | "
                 f"{f3(*rel_val(scheme, e, 'auc'))} | {f3(*rel_val(scheme, e, 'tpr_at_1pct_fpr'))} | "
                 f"{f3(*rel_val(scheme, e, 'evidence_recall3'))} |")
L += ["\n## Step 6 - distinguishability game: mean AUC over targets [95% CI]\n"]
for att, title in attacks + [("logreg_passive", "logistic regression, passive"), ("rf_passive", "random forest, passive")]:
    L += [f"\n**{title}**\n", "| group size | " + " | ".join(ELAB) + " |", "|---" * (len(ELAB) + 1) + "|"]
    for m in ms:
        cells = []
        for e in eps_list:
            g = gm[(gm.attack == att) & (gm.m == m) & (gm.eps == e)].auc
            cells.append("-" if len(g) == 0 else f"{g.mean():.3f} [{g.mean() - 1.96 * g.std() / np.sqrt(len(g)):.2f}, "
                                                 f"{g.mean() + 1.96 * g.std() / np.sqrt(len(g)):.2f}]")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
L += ["\n## Step 8 - per-case explanation attacks (AUC [95% CI], TPR at 1% FPR)\n",
      "| target | statistic | threshold test | shadow-calibrated test | TPR@1%FPR (calibrated) |", "|---|---|---|---|---|"]
th = cm[cm.test == "threshold"].set_index(["target", "statistic"])
for t in TORDER:
    for st, _ in STAT:
        a, b = th.loc[(t, st)], lrt.loc[(t, st)]
        L.append(f"| {t} | {st} | {a.auc:.3f} [{a.auc_lo:.2f}, {a.auc_hi:.2f}] | "
                 f"{b.auc:.3f} [{b.auc_lo:.2f}, {b.auc_hi:.2f}] | {b.tpr_at_1pct_fpr:.3f} |")
(res / "SUMMARY_N2.md").write_text("\n".join(L) + "\n", encoding="utf-8")

reg = load_json(path_n2(cfg2, "models") / "registry.json")
save_json({"python": sys.version.split()[0], "platform": f"{platform.system()} {platform.machine()}",
           "libraries": {m: importlib.import_module(m).__version__ for m in
                         ["numpy", "pandas", "scipy", "sklearn", "torch", "captum", "flwr"]},
           "config_n2_sha256": sha256_file(ROOT / "config_n2.yaml"),
           "config_n4_sha256": sha256_file(ROOT / "config_n4.yaml"),
           "config_n1_sha256": sha256_file(ROOT / "config.yaml"),
           "models": {r["file"]: r["sha256"] for r in reg}}, res / "run_manifest_n2.json")
print("Wrote 3 figures to figures_n2/, results_n2/SUMMARY_N2.md and results_n2/run_manifest_n2.json")
