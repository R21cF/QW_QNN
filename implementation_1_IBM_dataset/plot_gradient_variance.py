"""Figure and table for the gradient-variance diagnostic (reads the two JSON files)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# thesis figure palette (implementation_3_algorithms/common.py), ordered so the
# gray baseline is adjacent only to orange (validated: CVD dE 9.2, normal 17.6)
BLUE, GREEN, ORANGE, NEUTRAL = "#2a78d6", "#1baf7a", "#eb6834", "#8a8985"
INK_2, GRID = "#52514e", "#e3e2dd"
STYLE = {  # colour follows the entity in every panel
    "walk, per-vertex coin (real)":    dict(color=BLUE,    marker="o", ls="-"),
    "walk, per-vertex coin (complex)": dict(color=ORANGE,  marker="s", ls="-"),
    "walk, per-step coin":             dict(color=GREEN,   marker="^", ls="-"),
    "hardware-efficient":              dict(color=NEUTRAL, marker="D", ls="--"),
}
DEPTH_NAME = {"walk, real coin": "walk, per-vertex coin (real)",
              "walk, complex coin": "walk, per-vertex coin (complex)",
              "hardware-efficient": "hardware-efficient"}
WIDTH_NAME = {"walk, per-vertex coin": "walk, per-vertex coin (real)",
              "walk, per-step coin": "walk, per-step coin",
              "hardware-efficient": "hardware-efficient"}
SHORT = {"walk, per-vertex coin (real)": "per-vertex (real)",
         "walk, per-vertex coin (complex)": "per-vertex (complex)",
         "walk, per-step coin": "per-step",
         "hardware-efficient": "hardware-eff."}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2})


def series(ax, x, y, lo, hi, name):
    st = STYLE[name]
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)
    ax.plot(x, y, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"],
            ms=5, mec="white", mew=1.0, label=SHORT[name])
    ax._endlabels = getattr(ax, "_endlabels", []) + [(x[-1], y[-1], SHORT[name])]


def place_labels(ax, min_gap=0.07):
    """Direct end labels, pushed apart vertically (axis fraction) when lines end close."""
    import matplotlib.transforms as mt
    lo, hi = np.log10(ax.get_ylim())
    items = sorted(ax._endlabels, key=lambda t: t[1])
    frac = [(np.log10(y) - lo) / (hi - lo) for _, y, _ in items]
    for i in range(1, len(frac)):
        frac[i] = max(frac[i], frac[i - 1] + min_gap)
    shift = frac[-1] - 0.97
    if shift > 0:                       # keep the top label inside the axes
        frac = [f - shift for f in frac]
    blend = mt.blended_transform_factory(ax.transData, ax.transAxes)
    tr = mt.offset_copy(blend, fig=ax.figure, x=6, y=0, units="points")
    for (x, _y, text), f in zip(items, frac):
        ax.text(x, f, text, transform=tr, va="center", fontsize=6.8, color="#0b0b0b")


def style_axes(ax, xlabel, title):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=7.8, loc="left", color="#0b0b0b")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    d = json.load(open("gradient_variance.json"))
    w = json.load(open("gradient_variance_width.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))

    ax = axes[0]
    for raw, name in DEPTH_NAME.items():
        r = [x for x in d["rows"] if x["ansatz"] == raw]
        series(ax, [x["depth"] for x in r], [x["var"] for x in r],
               [x["ci_lo"] for x in r], [x["ci_hi"] for x in r], name)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
    ax.set_xlim(0.8, 40)
    style_axes(ax, "depth (steps or layers)", "(a) line detection, 5 qubits")
    ax.set_ylabel(r"Var$[\partial \mathcal{L} / \partial w]$")
    ax.set_yticks([3e-3, 4e-3, 6e-3, 8e-3])
    ax.set_yticklabels([r"$3\times10^{-3}$", r"$4\times10^{-3}$",
                        r"$6\times10^{-3}$", r"$8\times10^{-3}$"])
    ax.minorticks_off()

    for ax, key, ci, title in ((axes[1], "var_local", "local_ci", "(b) grid graphs, local cost"),
                               (axes[2], "var_global", "global_ci", "(c) grid graphs, global cost")):
        for raw, name in WIDTH_NAME.items():
            r = [x for x in w["rows"] if x["ansatz"] == raw]
            series(ax, [x["qubits"] for x in r], [x[key] for x in r],
                   [x[ci][0] for x in r], [x[ci][1] for x in r], name)
        ax.set_xlim(4.6, 12.4)
        ax.set_xticks([5, 6, 7, 8, 9, 10, 11, 12]); ax.tick_params(axis="x", labelsize=6.8)
        style_axes(ax, "qubits ($N = 2^{q-2}$ vertices)", title)
        ax.set_ylabel(r"Var$[\partial C / \partial w]$")
    
    handles = []
    for name in STYLE:
        st = STYLE[name]
        handles.append(plt.Line2D([], [], color=st["color"], ls=st["ls"], lw=2,
                                  marker=st["marker"], ms=5, mec="white", mew=1.0,
                                  label=name.replace("walk, ", "walk ansatz, ")))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("fig_gradient_variance.pdf")
    fig.savefig("fig_gradient_variance.png", dpi=160)

    # table: width sweep
    fits = w["log2_slope_per_qubit"]
    rows = {}
    for x in w["rows"]:
        rows.setdefault((x["qubits"], x["grid"]), {})[x["ansatz"]] = x
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption[Gradient variance of the walk ansatz against register width]"
             r"{Variance of the gradient of the cost with respect to one ansatz parameter, over "
             + str(w["n_draws"]) + r" uniformly random parameter draws, on grid graphs of "
             r"increasing size at depth $T = R + C$. Local cost: $Z$ on the most significant "
             r"qubit; global cost: $Z$ on every qubit, the classifier's observable. The last "
             r"row is the least-squares slope of $\log_2 \mathrm{Var}$ per added qubit.}",
             r"\label{tab:gradvar}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{rrrrrrrr}", r"\hline",
             r" & & \multicolumn{3}{c}{local cost} & \multicolumn{3}{c}{global cost} \\",
             r"Grid & Qubits & per-vertex & per-step & HEA & per-vertex & per-step & HEA \\",
             r"\hline"]

    def sci(v):
        m, e = f"{v:.1e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    order = ["walk, per-vertex coin", "walk, per-step coin", "hardware-efficient"]
    for (q, g), r in sorted(rows.items()):
        cells = [sci(r[a]["var_local"]) for a in order] + [sci(r[a]["var_global"]) for a in order]
        gt = g.replace("x", r"$\times$")
        lines.append(gt + f" & {q} & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{2}{l}{slope} & "
                 + " & ".join(f"${fits[a + ' var_local']:+.2f}$" for a in order) + " & "
                 + " & ".join(f"${fits[a + ' var_global']:+.2f}$" for a in order) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open("tab_gradvar.tex", "w").write("\n".join(lines) + "\n")
    print("wrote fig_gradient_variance.{pdf,png}, tab_gradvar.tex")


if __name__ == "__main__":
    main()
