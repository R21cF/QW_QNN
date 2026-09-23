"""
Reads the four results JSONs and writes fig_*.pdf and tab_*.tex; nothing is
computed here. Style matches implementation_1's plot scripts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RC, INK, INK_2, GRID, BLUE, ORANGE, GREEN, NEUTRAL, SURFACE

plt.rcParams.update(RC)


def save(fig, name):
    fig.savefig(name + ".pdf", bbox_inches="tight")
    fig.savefig(name + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ search
S = json.load(open("search_results.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
Ns = [r["N"] for r in S["dtqw"]]; ts = [r["t_peak"] for r in S["dtqw"]]
ax.plot(Ns, ts, "o-", color=BLUE, markersize=5, linewidth=1.6, label="coined DTQW, first peak")
Ns2 = [r["N"] for r in S["ctqw"]]; ts2 = [r["t_peak"] for r in S["ctqw"]]
ax.plot(Ns2, ts2, "s-", color=ORANGE, markersize=4.5, linewidth=1.6, label="CTQW, first peak")
xx = np.array([4, 1024.0])
ax.plot(xx, np.pi / 2 * np.sqrt(xx), "--", color=NEUTRAL, linewidth=1.0, label=r"$(\pi/2)\sqrt{N}$")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("$N$ (vertices of $K_N$)"); ax.set_ylabel("time to first success peak")
fa, fb = S["dtqw_fit"], S["ctqw_fit"]
ax.text(0.03, 0.97, f"DTQW fit $t \\sim N^{{{fa['alpha']:.2f}}}$\nCTQW fit $t \\sim N^{{{fb['alpha']:.2f}}}$",
        transform=ax.transAxes, va="top", fontsize=7.5, color=INK_2)
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.grid(color=GRID, linewidth=0.6, zorder=0)
ax.set_title("(a) linear walks", fontsize=8.5, color=INK, loc="left")

ax = axes[1]
cols = {0.0: NEUTRAL, 0.5: BLUE, 1.0: ORANGE, 2.0: GREEN}
for rows in S["nonlinear"]:
    g = rows[0]["g_over_N"]
    Ns = [r["N"] for r in rows]
    ax.plot(Ns, [r["t_peak"] for r in rows], "o-", color=cols[g], markersize=4, linewidth=1.4,
            label=f"$g = {g:g}N$" if g > 0 else "$g = 0$ (linear)")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("$N$"); ax.set_ylabel("time of first peak")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.grid(color=GRID, linewidth=0.6, zorder=0)
ax.set_title("(b) nonlinear CTQW, $g \\propto N$", fontsize=8.5, color=INK, loc="left")
save(fig, "fig_search_scaling")

# peak width and required g
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax = axes[0]
for rows in S["nonlinear"]:
    g = rows[0]["g_over_N"]
    if g == 0:
        continue
    ax.plot([r["N"] for r in rows], [r["fwhm"] for r in rows], "o-", color=cols[g], markersize=4,
            linewidth=1.4, label=f"$g = {g:g}N$")
xx = np.array([16, 1024.0])
ax.plot(xx, 3.5 / np.sqrt(xx), "--", color=NEUTRAL, linewidth=1.0, label=r"$\propto N^{-1/2}$")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("$N$"); ax.set_ylabel("FWHM of the success peak")
ax.legend(frameon=False, fontsize=7)
ax.grid(color=GRID, linewidth=0.6, zorder=0)
ax.set_title("(a) the peak narrows", fontsize=8.5, color=INK, loc="left")
ax = axes[1]
rows = S["g_required"][0]
Ns = [r["N"] for r in rows if r["g_min"]]; gs = [r["g_min"] for r in rows if r["g_min"]]
ax.plot(Ns, gs, "o-", color=BLUE, markersize=4.5, linewidth=1.6, label="$g_{\\min}$ for $p \\geq 1/2$ by $t=2$")
ft = S["g_fit_t2"]
xx = np.array([16, 512.0])
ax.plot(xx, ft["c"] * xx ** ft["alpha"], "--", color=NEUTRAL, linewidth=1.0,
        label=f"fit $g \\sim N^{{{ft['alpha']:.2f}}}$")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
ax.set_xlabel("$N$"); ax.set_ylabel("required nonlinearity $g$")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.grid(color=GRID, linewidth=0.6, zorder=0)
ax.set_title("(b) what constant time costs", fontsize=8.5, color=INK, loc="left")
save(fig, "fig_search_nonlinear")

# ------------------------------------------------------------------ shor
R = json.load(open("shor_results.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax = axes[0]
r15 = [r for r in R if r["N"] == 15 and r["a"] == 7][0]
ret = r15["return_prob"]
ax.bar(range(len(ret)), ret, color=BLUE, width=0.6, zorder=3)
ax.set_xlabel("walk steps $t$"); ax.set_ylabel(r"$|\langle 1|U_a^t|1\rangle|^2$")
ax.set_title("(a) revival of the walk, $N=15$, $a=7$, $r=4$", fontsize=8.5, color=INK, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
ax = axes[1]
m = "8"
q = r15["qpe"][m]
ks = np.zeros(2 ** int(m))
for o in q["outcomes"]:
    ks[o["k"]] = o["count"]
ks /= ks.sum()
ax.bar(np.arange(len(ks)) / len(ks), ks, width=0.012, color=BLUE, zorder=3)
for j in range(r15["r"]):
    ax.axvline(j / r15["r"], color=ORANGE, linewidth=0.8, linestyle="--", zorder=2)
ax.set_xlabel("measured phase $k/2^m$"); ax.set_ylabel("frequency (4096 shots)")
ax.set_title(f"(b) phase estimation on $U_a$, $m={m}$", fontsize=8.5, color=INK, loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
save(fig, "fig_shor_walk")

with open("tab_shor.tex", "w") as f:
    f.write("% Generated by make_figures.py -- do not edit by hand.\n")
    f.write("\\begin{table}[htbp]\n\\centering\\small\n")
    f.write("\\caption{Order finding as phase estimation on the permutation walk $U_a$. "
            "Cycle lengths are those of $x \\mapsto ax \\bmod N$ on the units of $\\mathbb{Z}_N$; "
            "$P(r)$ is the fraction of 4096 shots whose continued-fraction denominator is exactly $r$ "
            "(the remainder return divisors of $r$, as in Shor's analysis); depth and CX are for the "
            "transpiled circuit with $m = 2n$ counting qubits, basis $\\{\\mathrm{CX}, R_Z, \\sqrt{X}, X\\}$, "
            "with the controlled walk steps synthesised as dense unitaries.}\n")
    f.write("\\label{tab:shor}\n\\begin{tabular}{rrrlrrrrl}\n\\hline\n")
    f.write("$N$ & $a$ & $r$ & cycles & $m$ & $P(r)$ & qubits & CX & factors \\\\\n\\hline\n")
    for r in R:
        m = str(2 * r["n"])
        q = r["qpe"][m]
        cyc = ", ".join(str(c) for c in r["cycle_lengths"])
        fac = "; ".join(q["factors_found"]).replace("(", "").replace(")", "").replace(", ", "$\\times$") or "---"
        f.write(f"{r['N']} & {r['a']} & {r['r']} & {cyc} & {m} & {q['p_correct_r']:.2f} & "
                f"{q['qubits']} & {q['cx']} & {fac} \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ qaoa
Q = json.load(open("qaoa_results.json"))
ps = sorted(int(p) for p in Q[0]["results"])
fig, ax = plt.subplots(figsize=(7.2, 3.2))
names = [("CTQW-J", BLUE, "CTQW mixer on $J(8,4)$"), ("DTQW-J", ORANGE, "DTQW mixer on $J(8,4)$ (one step/layer)"),
         ("X", NEUTRAL, "$X$ mixer + penalty")]
w = 0.26
for i, (key, col, lab) in enumerate(names):
    mu = [np.mean([r["results"][str(p)][key]["ratio"] for r in Q]) for p in ps]
    sd = [np.std([r["results"][str(p)][key]["ratio"] for r in Q]) for p in ps]
    x = np.array(ps) + (i - 1) * w
    ax.bar(x, mu, width=w - 0.03, color=col, zorder=3, label=lab)
    ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=2, zorder=4)
    for xi, m_, s_ in zip(x, mu, sd):
        ax.text(xi, m_ + s_ + 0.015, f"{m_:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
base = np.mean([r["ratio_uniform_feasible"] for r in Q])
ax.axhline(base, color=INK, linewidth=0.9, linestyle=":", zorder=2)
ax.text(ps[0] - 0.45, base + 0.012, "uniform feasible", ha="left", va="bottom", fontsize=7, color=INK, style="italic")
ax.set_xticks(ps); ax.set_xlabel("QAOA depth $p$")
ax.set_ylabel(r"$\mathbb{E}[C(x)\,\mathbb{1}_{\mathrm{feasible}}]/C_{\mathrm{opt}}$")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
save(fig, "fig_qaoa_mixers")

with open("tab_qaoa.tex", "w") as f:
    f.write("% Generated by make_figures.py -- do not edit by hand.\n")
    f.write("\\begin{table}[htbp]\n\\centering\\small\n")
    f.write("\\caption{Balanced Max-Cut on six random weighted graphs ($n=8$, $|S|=4$): approximation ratio "
            "$\\mathbb{E}[C(x)\\mathbf{1}_{\\mathrm{feasible}}]/C_{\\mathrm{opt}}$ and probability of a feasible "
            "outcome, mean $\\pm$ s.d.\\ over instances, best of twelve COBYLA starts. A uniform draw from the "
            f"feasible set scores {base:.2f}.}}\n")
    f.write("\\label{tab:qaoa}\n\\begin{tabular}{clrr}\n\\hline\n")
    f.write("$p$ & Mixer & Ratio & $P(\\mathrm{feasible})$ \\\\\n\\hline\n")
    for p in ps:
        for key, _, lab in names:
            rr = [r["results"][str(p)][key]["ratio"] for r in Q]
            pf = [r["results"][str(p)][key]["p_feasible"] for r in Q]
            f.write(f"{p if key == 'CTQW-J' else ''} & {lab} & ${np.mean(rr):.3f} \\pm {np.std(rr):.3f}$ & "
                    f"${np.mean(pf):.2f} \\pm {np.std(pf):.2f}$ \\\\\n")
        f.write("\\hline\n")
    f.write("\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ graphs
try:
    Gr = json.load(open("graph_results.json"))
except FileNotFoundError:
    Gr = None
if Gr:
    methods = [("DTQW signature kernel", BLUE), ("classical RW signature kernel", ORANGE),
               ("WL subtree kernel", GREEN), ("hand-made statistics", NEUTRAL),
               ("walk network (trained coins)", "#7a4fd6"), ("walk network (fixed Grover coin)", "#c7b3f0")]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    tasks = [r["task"] for r in Gr]
    w = 0.13
    for i, (m_, col) in enumerate(methods):
        pts = [(t, 100 * r[m_][0], 100 * r[m_][1]) for t, r in enumerate(Gr) if m_ in r]
        x = np.array([t for t, _, _ in pts]) + (i - 2.5) * w
        mu = [v for _, v, _ in pts]; sd = [e for _, _, e in pts]
        ax.bar(x, mu, width=w - 0.02, color=col, zorder=3, label=m_)
        ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=1.5, zorder=4)
    for t, r in enumerate(Gr):
        maj = 100 * max(r["class_balance"], 1 - r["class_balance"])
        ax.plot([t - 0.42, t + 0.42], [maj, maj], color=INK, linewidth=0.9, linestyle=":", zorder=5)
    ax.plot([], [], color=INK, linewidth=0.9, linestyle=":", label="majority class")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, fontsize=7.5)
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    save(fig, "fig_graph_acc")
    with open("tab_graph.tex", "w") as f:
        f.write("% Generated by make_figures.py -- do not edit by hand.\n")
        f.write("\\begin{table}[htbp]\n\\centering\\small\n")
        f.write("\\caption{Graph classification on three synthetic tasks and four public benchmarks: test accuracy (\\%), mean $\\pm$ s.d.\\ "
                "over $3 \\times 10$ stratified folds. The WL kernel uses node labels where the dataset has them; the walks are label-blind. "
                "The walk network is not run on PROTEINS and IMDB-BINARY (---); the near-trivial regular task, on which every method scores 99.5 or 100, is omitted. Kernels are RBF--SVMs on the walk signature "
                "($T=8$ steps, four statistics per step) or the WL subtree kernel ($h=3$); the walk network "
                "is the same walk with one trainable coin angle per step and a logistic read-out.}\n")
        short = {"DTQW signature kernel": "DTQW signature", "classical RW signature kernel": "classical RW signature",
                 "WL subtree kernel": "WL subtree", "hand-made statistics": "hand-made statistics",
                 "walk network (trained coins)": "walk network, trained coins",
                 "walk network (fixed Grover coin)": "walk network, Grover coin"}
        Gt = [r for r in Gr if r["task"] != "regular"]          # trivial task: figure only
        col = {"PTC_MR": "PTC\\_MR", "IMDB-BINARY": "IMDB-B"}
        f.write("\\label{tab:graph}\n\\setlength{\\tabcolsep}{2.5pt}\\scriptsize\n\\begin{tabular}{l" + "r" * len(Gt) + "}\n\\hline\n")
        f.write("Method & " + " & ".join(col.get(r["task"], r["task"]) for r in Gt) + " \\\\\n\\hline\n")
        for m_, _ in methods:
            f.write(short[m_] + " & " + " & ".join(f"${100*r[m_][0]:.1f} \\pm {100*r[m_][1]:.1f}$" if m_ in r else "---" for r in Gt) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
print("done")
