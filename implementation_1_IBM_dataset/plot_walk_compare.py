"""
Classical symmetric random walk versus the Hadamard quantum walk on the line.

Produces fig_walk_compare.pdf/.png for Chapter 2 of the thesis and prints the
numbers quoted there. Exact arithmetic throughout (no sampling): the classical
distribution is the binomial law, the quantum one is the coined walk iterated
in NumPy with the symmetric initial coin state (|0> + i|1>)/sqrt(2). Needs only
NumPy and matplotlib, so it runs under code/.venv.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
CLASSICAL = "#eb6834"
QUANTUM = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

T = 200                      # steps
X = np.arange(-T, T + 1)     # sites reachable in T steps


def classical(T):
    """p_t(x) for t = 0..T, symmetric +-1 steps from the origin."""
    P = np.zeros((T + 1, 2 * T + 1))
    P[0, T] = 1.0
    for t in range(1, T + 1):
        P[t, 1:] += 0.5 * P[t - 1, :-1]
        P[t, :-1] += 0.5 * P[t - 1, 1:]
    return P


def hadamard(T, coin0=(1 / np.sqrt(2), 1j / np.sqrt(2))):
    """p_t(x) for the Hadamard walk, U = S (I x H), coin |0> moves right."""
    psi = np.zeros((2, 2 * T + 1), dtype=complex)        # [coin, x]
    psi[0, T], psi[1, T] = coin0
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    P = np.zeros((T + 1, 2 * T + 1))
    P[0] = (np.abs(psi) ** 2).sum(axis=0)
    for t in range(1, T + 1):
        c = h @ psi                                        # coin toss
        new = np.zeros_like(psi)
        new[0, 1:] = c[0, :-1]                             # |0>: x -> x+1
        new[1, :-1] = c[1, 1:]                             # |1>: x -> x-1
        psi = new
        P[t] = (np.abs(psi) ** 2).sum(axis=0)
    return P


Pc = classical(T)
Pq = hadamard(T)
assert np.allclose(Pc.sum(axis=1), 1) and np.allclose(Pq.sum(axis=1), 1)

mean_c = Pc @ X
mean_q = Pq @ X
sig_c = np.sqrt(Pc @ X**2 - mean_c**2)
sig_q = np.sqrt(Pq @ X**2 - mean_q**2)
ts = np.arange(T + 1)
slope = np.sqrt(1 - 1 / np.sqrt(2))
xx = X[X % 2 == 0]

# limit laws for the comparison at t = T (even sites carry all the mass)
t0 = 100
even = (X % 2 == 0)
xs = X[even] / t0
gauss = np.exp(-xs**2 * t0 / 2) / np.sqrt(2 * np.pi / t0) * 2 / t0   # density x (site spacing 2)/t0
inside = np.abs(xs) < 1 / np.sqrt(2) - 1e-9
konno = np.zeros_like(xs)
konno[inside] = 1 / (np.pi * (1 - xs[inside]**2) * np.sqrt(1 - 2 * xs[inside]**2))
konno *= 2 / t0

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.0),
                         gridspec_kw={"hspace": 0.38, "height_ratios": [1.5, 1.0]})

# (a): distributions at t = t0 on the even sub-lattice, with the limit laws
ax = axes[0]
ax.plot(xx, Pc[t0][even], color=CLASSICAL, lw=1.7, label="classical, $t=100$")
ax.plot(xx, Pq[t0][even], color=QUANTUM, lw=1.7, label="Hadamard, $t=100$")
ax.plot(xx, gauss, color=INK, lw=0.8, ls="--", label="Gaussian, $\\sigma=\\sqrt{t}$")
ax.plot(xx, konno, color=INK, lw=0.8, ls=":", label="Konno limit law")
ax.set_xlim(-100, 100)
ax.set_ylim(0, 0.115)
ax.set_xlabel("position $x$")
ax.set_ylabel("$p_t(x)$ (even sites)")
ax.set_title("(a) position distribution after 100 steps", loc="left", color=INK, fontsize=9.5)
ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.8)

# (b): spread
ax = axes[1]
ax.plot(ts, sig_c, color=CLASSICAL, lw=2.2, alpha=0.5, label="classical")
ax.plot(ts, sig_q, color=QUANTUM, lw=2.2, alpha=0.5, label="Hadamard")
ax.plot(ts, np.sqrt(ts), color=INK, lw=0.8, ls="--", label="$\\sqrt{t}$")
slope = np.sqrt(1 - 1 / np.sqrt(2))
ax.plot(ts, slope * ts, color=INK, lw=0.8, ls=":", label="$\\sqrt{1-1/\\sqrt{2}}\\; t$")
ax.set_xlim(0, T)
ax.set_ylim(0, 120)
ax.set_xlabel("step $t$")
ax.set_ylabel("standard deviation $\\sigma_t$")
ax.set_title("(b) spread of the walker", loc="left", color=INK, fontsize=9.5)
ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6)

for ax in axes:
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.savefig("fig_walk_compare.pdf", bbox_inches="tight")
fig.savefig("fig_walk_compare.png", dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------- numbers for the text
def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

out = {
    "T": T,
    "sigma_classical_t100": float(sig_c[100]), "sigma_hadamard_t100": float(sig_q[100]),
    "sigma_classical_t200": float(sig_c[200]), "sigma_hadamard_t200": float(sig_q[200]),
    "sigma_hadamard_over_t_at_200": float(sig_q[200] / 200),
    "konno_slope": float(slope),
    "max_p_classical_t100": float(Pc[100].max()), "max_p_hadamard_t100": float(Pq[100].max()),
    "argmax_hadamard_t100": [int(x) for x in X[Pq[100] > Pq[100].max() - 1e-12]],
    "entropy_bits_classical_t100": entropy(Pc[100]), "entropy_bits_hadamard_t100": entropy(Pq[100]),
    "mass_within_t_over_sqrt2_hadamard_t100": float(Pq[100][np.abs(X) <= 100 / np.sqrt(2)].sum()),
    "mass_within_2sqrt_t_classical_t100": float(Pc[100][np.abs(X) <= 2 * np.sqrt(100)].sum()),
    "return_prob_origin_classical_t100": float(Pc[100][T]), "return_prob_origin_hadamard_t100": float(Pq[100][T]),
}
json.dump(out, open("walk_compare.json", "w"), indent=2)
print(json.dumps(out, indent=2))
