"""N4 STEP 9 - Tables, figures and a one-page summary of YOUR N4 numbers.

Output: figures_n4/fig4_accuracy_and_ter_vs_alpha.png, fig5_class_rarity.png,
        fig6_poisoning.png, fig7_proxy_share.png, fig8_convergence.png,
        results_n4/SUMMARY_N4.md, results_n4/run_manifest_n4.json
Run:  python n4_07_make_report.py
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
from ufn4.common import load_config_n4, path_n4                     # noqa: E402

cfg4 = load_config_n4()
res, fig_dir, mdir = path_n4(cfg4, "results"), path_n4(cfg4, "figures"), path_n4(cfg4, "models")
E = "integrated_gradients"            # main explainer for N4 (validated in N1, Level 1)
s = pd.read_csv(res / "model_scores.csv", dtype={"method": str, "attack": str, "rule": str})
pc = pd.read_csv(res / "per_case_scores.csv.gz", dtype={"method": str, "attack": str, "rule": str},
                 low_memory=False)
alphas = cfg4["alphas"]
ALAB = {0.1: "0.1\nvery skewed", 0.5: "0.5", 1.0: "1.0", 100.0: "100\n~ IID"}
COL = {"Centralised": "#2a78d6", "FedAvg": "#eb6834", "FedProx": "#1baf7a", "Local-only": "#6250d6"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def setting_stats(metric):
    """mean and SD per setting and alpha. Federated/central: SD over seeds.
    Local-only: per seed the mean of the 8 agencies, then mean/SD over seeds;
    plus min/max over all individual agency models."""
    out = {}
    c = s[s.kind == "central"][metric]
    for a in alphas:
        out[("Centralised", a)] = (c.mean(), c.std())
        for m, lab in [("fedavg", "FedAvg"), ("fedprox", "FedProx")]:
            v = s[(s.kind == "federated") & (s.method == m) & (s.alpha == a)][metric]
            out[(lab, a)] = (v.mean(), v.std())
        l = s[(s.kind == "local") & (s.alpha == a)]
        per_seed = l.groupby("seed")[metric].mean()
        out[("Local-only", a)] = (per_seed.mean(), per_seed.std(), l[metric].min(), l[metric].max())
    return out


# ------------------------------------------------ Fig 4: macro-F1 and TER vs alpha
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, metric, title in [(axes[0], "macro_f1", "Triage macro-F1"),
                          (axes[1], f"ter_{E}", "Trustworthy-Explanation Rate (TER)")]:
    st = setting_stats(metric)
    x = np.arange(len(alphas))
    for k, lab in enumerate(COL):
        mu = [st[(lab, a)][0] for a in alphas]
        sd = [st[(lab, a)][1] for a in alphas]
        ax.errorbar(x + (k - 1.5) * 0.06, mu, yerr=sd, color=COL[lab], lw=2, marker="o", ms=5,
                    capsize=3, label=lab, ls="--" if lab == "Centralised" else "-")
    ax.set_xticks(x, [ALAB[a] for a in alphas])
    ax.set_xlabel("Dirichlet alpha (how different the agencies' case mixes are)")
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", color=INK)
    tidy(ax)
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Federation matches centralised training; agencies alone fall far behind",
             x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(fig_dir / "fig4_accuracy_and_ter_vs_alpha.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 5: class rarity at an agency
reg = {r["file"]: r for r in load_json(mdir / "registry_baselines.json")}
loc = pc[pc.kind == "local"].copy()
loc["n_class"] = [reg[f]["train_counts"][lab] for f, lab in zip(loc.file, loc.label)]
loc["good"] = loc.correct & (loc[f"recall_{E}"] >= cfg4["scoring"]["good_explanation_recall"])
bins, blabs = [-1, 0, 9, 29, 99, 10 ** 6], ["0", "1-9", "10-29", "30-99", "100+"]
loc["bucket"] = pd.cut(loc.n_class, bins, labels=blabs)
rar = loc.groupby("bucket", observed=False).agg(cases=("correct", "size"), models=("file", "nunique"),
                                                detected=("correct", "mean"), ter=("good", "mean"),
                                                proxy=(f"proxy_{E}", "mean")).reset_index()
fed = pc[(pc.kind == "federated") & (pc.method == "fedavg")]
fed_ter = float((fed.correct & (fed[f"recall_{E}"] >= cfg4["scoring"]["good_explanation_recall"])).mean())
fed_det = float(fed.correct.mean())
fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(blabs))
ax.bar(x - 0.18, rar.detected, 0.34, color="#2a78d6", label="local model detects the crime")
ax.bar(x + 0.18, rar.ter, 0.34, color="#eb6834", label="... and its explanation finds the evidence (TER)")
for i, (d, t) in enumerate(zip(rar.detected, rar.ter)):
    ax.text(i - 0.18, d + 0.015, f"{d:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(i + 0.18, t + 0.015, f"{t:.2f}", ha="center", fontsize=8, color=INK)
ax.axhline(fed_ter, color=INK, ls="--", lw=1)
ax.text(-0.4, fed_ter + 0.02, f"FedAvg TER {fed_ter:.2f}", ha="left", fontsize=9, color=INK)
ax.set_xticks(x, blabs)
ax.set_xlabel("Cases of that crime type the agency trained on")
ax.set_ylim(0, 1.15)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=9)
ax.set_title("An agency cannot learn - or explain - crimes it rarely sees", loc="left", color=INK)
tidy(ax)
fig.tight_layout()
fig.savefig(fig_dir / "fig5_class_rarity.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 6: poisoning
P = s[s.kind == "poison"]
rules = cfg4["poisoning"]["aggregators"]
attacks = cfg4["poisoning"]["attacks"]
RLAB = {"fedavg": "FedAvg", "fedmedian": "FedMedian", "fedtrimmedavg": "FedTrimmedAvg", "krum": "Krum"}
ALAB2 = {"none": "no attack", "label_flip": "label flip (fraud -> benign)", "scaled_update": "scaled update (x-5)"}
ACOL = {"none": "#2a78d6", "label_flip": "#eb6834", "scaled_update": "#1baf7a"}
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(rules))
for k, at in enumerate(attacks):
    g = P[P.attack == at].groupby("rule")[f"ter_{E}"]
    ax.bar(x + (k - 1) * 0.27, g.mean().reindex(rules), 0.25, color=ACOL[at], label=ALAB2[at])
    for j, r in enumerate(rules):                       # each seed as a dot: n = 3 is too few for SD bars
        v = P[(P.attack == at) & (P.rule == r)][f"ter_{E}"].to_numpy()
        ax.scatter(np.full(len(v), j + (k - 1) * 0.27), v, s=14, color=INK, zorder=3,
                   label="one seed" if (k == 0 and j == 0) else None)
ax.set_xticks(x, [RLAB[r] for r in rules])
ax.set_ylabel("TER (bar = mean of 3 seeds)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9)
ax.set_title("One malicious agency: robust rules keep the explanations right", loc="left", color=INK)
tidy(ax)
fig.tight_layout()
fig.savefig(fig_dir / "fig6_poisoning.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 7: proxy share
groups = [("Centralised", s[s.kind == "central"]), ("FedAvg", s[(s.kind == "federated") & (s.method == "fedavg")]),
          ("FedProx", s[(s.kind == "federated") & (s.method == "fedprox")]), ("Local-only", s[s.kind == "local"])]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
vals = [g[f"proxy_share_{E}"].mean() for _, g in groups]
sds = [g[f"proxy_share_{E}"].std() for _, g in groups]
ax.bar(range(4), vals, 0.55, yerr=sds, capsize=3, color=[COL[n] for n, _ in groups],
       error_kw={"ecolor": INK2, "elinewidth": 1})
for i, v in enumerate(vals):
    ax.text(i, v + sds[i] + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(range(4), [n for n, _ in groups])
ax.set_ylabel("share of top features that are proxies")
ax.set_title("Alone, agencies' models lean on indirect clues", loc="left", color=INK)
tidy(ax)
fig.tight_layout()
fig.savefig(fig_dir / "fig7_proxy_share.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 8: convergence
conv = load_json(res / "convergence.json")
fig, ax = plt.subplots(figsize=(7, 3.8))
for k, a in enumerate(alphas):
    h = np.array([conv[f"fedavg|{a}|{sd}"] for sd in cfg4["run_seeds"]])
    ax.plot(np.arange(1, h.shape[1] + 1), h.mean(0), lw=2, color=list(COL.values())[k],
            label=f"alpha {a:g}")
ax.set_xlabel("communication round")
ax.set_ylabel("validation macro-F1 (FedAvg)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="lower right")
ax.set_title("FedAvg converges within ~10 rounds except at extreme skew", loc="left", color=INK)
tidy(ax)
fig.tight_layout()
fig.savefig(fig_dir / "fig8_convergence.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ SUMMARY_N4.md
f3 = lambda m, sd=None: f"{m:.3f}" if sd is None or np.isnan(sd) else f"{m:.3f} +/- {sd:.3f}"
L = ["# N4 results summary (all numbers from this run)\n",
     f"Seeds: {cfg4['run_seeds']}. {cfg4['agencies']} agencies. Test set = N1's 600 cases. "
     f"Explainer for TER/recall/proxy: {E}.\n"]
cx = load_json(res / "case_export_check.json")
L += ["## CASE/UCO check\n",
      f"- {cx['cases']} cases exported ({cx['objects']} CASE objects); feature mismatches after round trip: "
      f"{len(cx['feature_mismatches'])}",
      f"- Official validation ({cx['case_version']}): " +
      ", ".join(f"{v['case_id']} conforms={v['conforms']}" for v in cx["validation"]) + "\n"]
for metric, name in [("macro_f1", "Macro-F1"), (f"ter_{E}", "TER"), (f"recall_{E}", "Evidence recall@|G| (oracle-passed)"),
                     ("oracle_pass_rate", "Oracle pass rate")]:
    st = setting_stats(metric)
    L += [f"## {name} by alpha (mean +/- SD over seeds)\n",
          "| alpha | Centralised | FedAvg | FedProx | Local-only mean [min, max] |", "|---|---|---|---|---|"]
    for a in alphas:
        lo = st[("Local-only", a)]
        L.append(f"| {a:g} | {f3(*st[('Centralised', a)])} | {f3(*st[('FedAvg', a)])} | "
                 f"{f3(*st[('FedProx', a)])} | {f3(lo[0], lo[1])} [{lo[2]:.3f}, {lo[3]:.3f}] |")
    L.append("")
L += ["## Local-only models: crime types rarely seen at the agency\n",
      "| Cases of that type at the agency | test cases | models | detected | TER | proxy share |",
      "|---|---|---|---|---|---|"]
for _, r in rar.iterrows():
    L.append(f"| {r.bucket} | {r.cases} | {r.models} | {r.detected:.3f} | {r.ter:.3f} | {r.proxy:.3f} |")
L.append(f"\nFedAvg (all alphas, all seeds): detected {fed_det:.3f}, TER {fed_ter:.3f}\n")
L += ["## Proxy share (correctly classified offence cases)\n", "| Setting | mean | SD | models |", "|---|---|---|---|"]
for (n, g), v, sd in zip(groups, vals, sds):
    L.append(f"| {n} | {v:.3f} | {sd:.3f} | {len(g)} |")
pr = load_json(mdir / "registry_poisoning.json")
shares = sorted({(r["seed"], r["attacker"], round(100 * r["attacker_share"])) for r in pr})
L += ["\n## Poisoning (alpha {}, mean +/- SD over seeds)\n".format(cfg4["poisoning"]["alpha"]),
      "Attacker per seed: " + "; ".join(f"seed {a}: agency {b} ({c}% of cases)" for a, b, c in shares) + "\n",
      "| Rule | Attack | Macro-F1 | Fraud F1 | TER |", "|---|---|---|---|---|"]
for rule in rules:
    for at in attacks:
        g = P[(P.rule == rule) & (P.attack == at)]
        L.append(f"| {RLAB[rule]} | {ALAB2[at]} | {f3(g.macro_f1.mean(), g.macro_f1.std())} | "
                 f"{f3(g.f1_fraud.mean(), g.f1_fraud.std())} | {f3(g[f'ter_{E}'].mean(), g[f'ter_{E}'].std())} |")
L += ["\n## Convergence (FedAvg, validation macro-F1)\n", "| alpha | round reaching 90% of best | final |", "|---|---|---|"]
for a in alphas:
    h = np.array([conv[f"fedavg|{a}|{sd}"] for sd in cfg4["run_seeds"]]).mean(0)
    L.append(f"| {a:g} | {int(np.argmax(h >= 0.9 * h.max())) + 1} | {h[-1]:.3f} |")
(res / "SUMMARY_N4.md").write_text("\n".join(L) + "\n", encoding="utf-8")

manifest = {"python": sys.version.split()[0], "platform": f"{platform.system()} {platform.machine()}",
            "libraries": {m: importlib.import_module(m).__version__ for m in
                          ["numpy", "pandas", "torch", "captum", "flwr", "case_utils", "sklearn"]},
            "config_n4_sha256": sha256_file(ROOT / "config_n4.yaml"),
            "config_n1_sha256": sha256_file(ROOT / "config.yaml"),
            "models": {r["file"]: r["sha256"] for part in ("baselines", "federated", "poisoning")
                       for r in load_json(mdir / f"registry_{part}.json")}}
save_json(manifest, res / "run_manifest_n4.json")
print("Wrote 5 figures to figures_n4/, results_n4/SUMMARY_N4.md and results_n4/run_manifest_n4.json")
