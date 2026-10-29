"""STEP 13 - Paper-ready figures and a one-page summary of YOUR numbers.

Output: figures/fig1_recall_by_tier.png, figures/fig2_disagreement_FA.png,
        figures/fig3_ffidelity.png, results/SUMMARY.md
Run:  python 11_make_report.py
"""
import matplotlib
matplotlib.use("Agg")                                  # no display needed (works on servers)
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from ufn1.common import get_path, load_config, load_json   # noqa: E402
from ufn1.explainers import EXPLAINERS                    # noqa: E402

cfg = load_config()
res, fig_dir = get_path(cfg, "results"), get_path(cfg, "figures")
tiers = cfg["benchmark"]["tiers"]
order = EXPLAINERS + ["treeshap_xgb"]
LABEL = {"kernelshap": "KernelSHAP", "integrated_gradients": "Integrated Gradients",
         "gradientshap": "GradientSHAP", "lime": "LIME", "feature_ablation": "Feature Ablation",
         "random": "Random (floor)", "treeshap_xgb": "TreeSHAP (XGBoost ref.)"}
TIER_COLOR = {"easy": "#2a78d6", "mixed": "#eb6834", "subtle": "#1baf7a"}   # validated palette slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})

# ------------------------------------------------ Fig 1: recall by explainer x tier
gt = pd.read_csv(res / "table_groundtruth.csv")
fig, ax = plt.subplots(figsize=(8.2, 5.0))
h = 0.26
for t_i, tier in enumerate(tiers):
    r = gt[gt.tier == tier].set_index("explainer").reindex(order)
    ypos = np.arange(len(order)) + (t_i - 1) * h
    err = np.vstack([r["recall@|G|"] - r["recall@|G|_lo"], r["recall@|G|_hi"] - r["recall@|G|"]])
    ax.barh(ypos, r["recall@|G|"], height=h - 0.04, color=TIER_COLOR[tier], label=tier,
            xerr=err, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
ax.axvline(cfg["acceptance"]["method_min_recall"], color=INK, lw=1, ls="--")
ax.text(cfg["acceptance"]["method_min_recall"] - 0.01, len(order) - 0.45, "method threshold",
        color=INK, fontsize=8, ha="right", va="center")
ax.set_yticks(np.arange(len(order)), [LABEL[e] for e in order])
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("Evidence recall@|G|  (share of planted features found; 95% CI)")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(title="tier", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_title("Do explanations point at the planted evidence?", loc="left", color=INK)
fig.tight_layout()
fig.savefig(fig_dir / "fig1_recall_by_tier.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 2: disagreement heatmap (FA@k)
M = pd.read_csv(res / "disagreement_FA.csv", index_col=0).reindex(index=EXPLAINERS, columns=EXPLAINERS)
fig, ax = plt.subplots(figsize=(7.4, 5.6))
im = ax.imshow(M.to_numpy(), cmap="Blues", vmin=0, vmax=1)
names = [LABEL[e] for e in EXPLAINERS]
ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        v = M.iat[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.046)
cb.set_label(f"Feature agreement of top-{cfg['scoring']['k_fixed']} (1 = same features)")
ax.set_title("How much do explainers agree with each other?", loc="left", color=INK)
fig.tight_layout()
fig.savefig(fig_dir / "fig2_disagreement_FA.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ Fig 3: F-Fidelity vs naive deletion
ff = pd.read_csv(res / "table_ffidelity.csv")
ff = ff[ff.tier == "ALL"].set_index("explainer").reindex(EXPLAINERS)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, col, title in [(axes[0], "ffid_plus", "F-Fidelity FFid+ (higher = more faithful)"),
                       (axes[1], "naive_deletion_drop", "Naive deletion: drop in P(class)")]:
    err = np.vstack([ff[col] - ff[col + "_lo"], ff[col + "_hi"] - ff[col]])
    ax.barh(range(len(ff)), ff[col], color="#2a78d6", height=0.6, xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(ff)), [LABEL[e] for e in EXPLAINERS])
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(fig_dir / "fig3_ffidelity.png", dpi=200)
plt.close(fig)

# ------------------------------------------------ SUMMARY.md
perf = load_json(res / "model_performance.json")
glob = load_json(res / "oracle_global.json")
orc = pd.DataFrame(load_json(res / "oracle_cases.json"))
acc = load_json(res / "acceptance_summary.json")
lines = ["# N1 results summary (all numbers from this run)\n",
         f"Seed: {cfg['seed']}. Test split: {sum(1 for _ in load_json(get_path(cfg, 'data') / 'test_answer_key.json'))} cases.\n",
         "## Triage models (test macro-F1)\n",
         f"- MLP: {perf['mlp']['macro_f1']:.3f}; XGBoost: {perf['xgb']['macro_f1']:.3f}",
         f"- MLP accuracy by tier: " + ", ".join(f"{k} {v:.3f}" for k, v in perf['mlp']['accuracy_by_tier'].items()),
         "\n## Oracle check\n",
         f"- MLP macro-F1: all features {glob['all_features']:.3f}; without decisive {glob['without_decisive']:.3f}; "
         f"without decisive and proxies {glob['without_decisive_and_proxies']:.3f}; "
         f"only decisive {glob['only_decisive']:.3f}",
         f"- Proxy features: {', '.join(glob['proxy_features'])}"]
for m in ["mlp", "xgb"]:
    o = orc[orc.model == m]
    lines.append(f"- {m.upper()} per-case pass rate: {o.passes.mean():.3f} "
                 + "(" + ", ".join(f"{t} {o[o.tier == t].passes.mean():.3f}" for t in tiers) + ")")
lines += ["\n## Evidence recall@|G| (oracle-passed cases, mean [95% CI])\n",
          "| Explainer | " + " | ".join(tiers) + " | all |", "|---" * (len(tiers) + 2) + "|"]
for e in order:
    cells = []
    for t in tiers + ["ALL"]:
        r = gt[(gt.explainer == e) & (gt.tier == t)].iloc[0]
        cells.append(f"{r['recall@|G|']:.3f} [{r['recall@|G|_lo']:.2f}, {r['recall@|G|_hi']:.2f}]")
    lines.append(f"| {LABEL[e]} | " + " | ".join(cells) + " |")
lines += ["\n## F-Fidelity (pooled, oracle-passed cases)\n",
          "| Explainer | FFid+ | FFid- | FFid+ (prob) | FFid- (prob) | naive deletion drop |",
          "|---|---|---|---|---|---|"]
for e, r in ff.iterrows():
    lines.append(f"| {LABEL[e]} | {r.ffid_plus:.3f} [{r.ffid_plus_lo:.3f}, {r.ffid_plus_hi:.3f}] | {r.ffid_minus:.3f} | "
                 f"{r.ffid_plus_prob:.3f} | {r.ffid_minus_prob:.3f} | {r.naive_deletion_drop:.3f} |")
lines += ["\n## Acceptance test\n",
          "Level 1 (method validation): " + ", ".join(
              f"{LABEL[e]} {'VALIDATED' if v['validated'] else 'not validated'}" for e, v in acc["level1"].items()),
          f"\nLevel 2 confidence cut-off calibrated on the validation split: tau = {acc['tau']}\n",
          "| Subset (test) | n | share good | validated | P(good \\| validated) | bad caught | good rejected |",
          "|---|---|---|---|---|---|---|"]
f3 = lambda x: "n/a" if x is None else f"{x:.3f}"
for k, v in acc["level2_test"].items():
    lines.append(f"| {k} | {v['n']} | {v['share_good']:.3f} | {v['validated_rate']:.3f} | "
                 f"{f3(v['precision_of_validated'])} | {f3(v['bad_caught_rate'])} | {f3(v['good_rejected_rate'])} |")
try:
    sc = load_json(res / "scaffold_attack.json")
    lines += ["\n## Scaffold attack (optional step)\n"]
    for e, v in sc["explainers"].items():
        lines.append(f"- {LABEL[e]}: recall {v['recall_before']:.3f} -> {v['recall_after']:.3f}; "
                     f"innocent feature ranked first in {v['innocent_feature_top1']:.3f} of cases")
    a = sc["agreement_check"]
    lines.append(f"- Agreement check flags {a['flag_rate_clean_model']:.3f} of honest-model cases and "
                 f"{a['flag_rate_scaffold']:.3f} of scaffold cases; fooled primary explanations caught: "
                 f"{f3(a['fooled_and_flagged'])}")
except FileNotFoundError:
    lines.append("\n(Scaffold attack not run.)")
(res / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------------------------------ run manifest (reproducibility)
import importlib, platform, sys                                         # noqa: E401,E402
from ufn1.common import ROOT, save_json, sha256_file                   # noqa: E402
manifest = {
    "python": sys.version.split()[0], "platform": f"{platform.system()} {platform.machine()}",
    "libraries": {m: importlib.import_module(m).__version__ for m in
                  ["numpy", "pandas", "scipy", "sklearn", "torch", "captum", "shap", "xgboost", "matplotlib"]},
    "config_sha256": sha256_file(ROOT / "config.yaml"),
    "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                     sorted((ROOT / cfg["paths"]["models"]).glob("*")) +
                     sorted((ROOT / cfg["paths"]["data"]).glob("*_answer_key.json"))},
}
save_json(manifest, res / "run_manifest.json")
print("Wrote results/run_manifest.json (versions + hashes, for reproducibility)")
print("Wrote figures/fig1_recall_by_tier.png, fig2_disagreement_FA.png, fig3_ffidelity.png")
print("Wrote results/SUMMARY.md")
