"""
Thesis figures for the QNN experiment.

Two figures, each answering one question:
  fig_qnn_accuracy  which architecture classifies better, and is any gap
                    larger than the seed-to-seed spread
  fig_qnn_training  how the optimisation behaves, which is what Sec. 5.2.1
                    ("Training Behaviour") reports

Colour carries one distinction only -- real coin against complex coin -- with
the non-walk baselines in neutral ink, because that contrast is the question
the experiment was built to answer.  Palette slots are the validated
categorical blue and orange; every bar is directly labelled, so identity never
rests on colour alone and the figures survive greyscale printing.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
REAL = "#2a78d6"
CPLX = "#eb6834"
NEUTRAL = "#8a8985"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

rows = json.load(open("qnn_results.json"))
by_name = {r["name"]: r for r in rows}

CLASSICAL_RBF = 100.0


def kind(name):
    if name.startswith("course"):
        return "baseline"
    # the coin family that carries the trainable weights is what the
    # real/complex question is about; a complex feature map with a real
    # ansatz is counted as complex because the encoding is where it differs
    return "complex" if "cplx" in name else "real"


# ---------------------------------------------------------------- fig 1 ---

order = sorted(rows, key=lambda r: r["test_acc_mean"])
labels = [r["name"].replace("course ZFeatureMap", "ZFeatureMap")
          .replace("walk ansatz", "walk ans.") for r in order]
means = np.array([r["test_acc_mean"] * 100 for r in order])
stds = np.array([r["test_acc_std"] * 100 for r in order])
colors = [{"baseline": NEUTRAL, "real": REAL, "complex": CPLX}[kind(r["name"])]
          for r in order]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
y = np.arange(len(order))
ax.barh(y, means, height=0.5, color=colors, linewidth=0, zorder=3)
ax.errorbar(means, y, xerr=stds, fmt="none", ecolor=INK_2, elinewidth=0.9,
            capsize=2.0, zorder=4)

ax.axvline(CLASSICAL_RBF, color=INK, linewidth=0.9, zorder=2)
ax.text(CLASSICAL_RBF - 1.2, len(order) - 0.35, "classical RBF-SVM: 100.0",
        ha="right", va="center", fontsize=7.5, color=INK, style="italic")

for i, (m, s, r) in enumerate(zip(means, stds, order)):
    ax.text(m + s + 1.2, i, f"{m:.1f}", va="center", ha="left",
            fontsize=8, color=INK)
    ax.text(1.8, i, f"{r['qubits']}q / {r['weights']}p", va="center", ha="left",
            fontsize=6.8, color=SURFACE, zorder=5)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_ylim(-0.8, len(order) - 0.1)
ax.set_xlim(0, 110)
ax.set_xlabel("test accuracy (%),  mean $\\pm$ s.d. over 10 weight initialisations")
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

handles = [plt.Line2D([], [], color=c, linewidth=5, label=l) for c, l in
           [(NEUTRAL, "course baseline (no walk)"), (REAL, "real coin"),
            (CPLX, "complex coin")]]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=3, frameon=False, fontsize=7.5)

fig.tight_layout()
fig.savefig("fig_qnn_accuracy.pdf", bbox_inches="tight")
fig.savefig("fig_qnn_accuracy.png", dpi=200, bbox_inches="tight")
print("wrote fig_qnn_accuracy.pdf / .png")


# ---------------------------------------------------------------- fig 2 ---

KEY = [
    ("course ZFeatureMap + full ansatz", NEUTRAL, "-"),
    ("walk FM (real) + walk ansatz (real, 4 steps)", REAL, "-"),
    ("walk FM (real) + walk ansatz (cplx)", CPLX, "-"),
    ("walk FM (real) + walk ansatz (real)", REAL, "--"),
]

fig, ax = plt.subplots(figsize=(7.2, 3.2))
for name, color, ls in KEY:
    hs = by_name[name]["histories"]
    L = min(len(h) for h in hs)
    arr = np.array([h[:L] for h in hs])
    med = np.median(arr, axis=0)
    lo, hi = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
    x = np.arange(L)
    ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
    lab = (name.replace("course ZFeatureMap + full ansatz",
                        "ZFeatureMap + full ansatz (8q, 16p)")
           .replace("walk FM (real) + walk ansatz (real, 4 steps)",
                    "walk FM + walk ansatz, real (5q, 32p)")
           .replace("walk FM (real) + walk ansatz (cplx)",
                    "walk FM + walk ansatz, complex (5q, 32p)")
           .replace("walk FM (real) + walk ansatz (real)",
                    "walk FM + walk ansatz, real (5q, 16p)"))
    ax.plot(x, med, color=color, linewidth=1.6, linestyle=ls, label=lab, zorder=3)

ax.set_xlabel("COBYLA objective evaluation")
ax.set_ylabel("training MSE")
ax.set_ylim(0, None)
ax.grid(color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.tight_layout()
fig.savefig("fig_qnn_training.pdf", bbox_inches="tight")
fig.savefig("fig_qnn_training.png", dpi=200, bbox_inches="tight")
print("wrote fig_qnn_training.pdf / .png")


# ------------------------------------------------------- LaTeX table ------

with open("tab_qnn_results.tex", "w") as f:
    f.write("% Generated by plot_qnn.py -- do not edit by hand.\n")
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Line-detection task: test accuracy over ten weight "
            "initialisations.}\n\\label{tab:qnn}\n")
    f.write("\\begin{tabular}{lrrr}\n\\hline\n")
    f.write("Model & Qubits & Params & Test acc.\\ (\\%) \\\\\n\\hline\n")
    for r in sorted(rows, key=lambda r: -r["test_acc_mean"]):
        nm = r["name"].replace("&", "\\&").replace("_", "\\_")
        f.write(f"{nm} & {r['qubits']} & {r['weights']} & "
                f"${r['test_acc_mean'] * 100:.1f} \\pm "
                f"{r['test_acc_std'] * 100:.1f}$ \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
print("wrote tab_qnn_results.tex")
