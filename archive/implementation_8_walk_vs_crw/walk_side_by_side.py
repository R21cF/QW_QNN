"""
Classical random walk against the Hadamard walk on the line, side by side.

Produces, for the introduction of the pre-defence presentation,
  fig_walk_side.pdf / .png   two panels at t = 100 on a common x-axis:
      (left)  symmetric +-1 random walk (binomial law)
      (right) Hadamard walk from the symmetric coin state (|0> + i|1>)/sqrt2

Exact arithmetic, nothing sampled; the walks are those of walk_vs_crw.py.
Only even sites are plotted (odd sites have probability zero at even t).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSICAL = "#c0504d"
QUANTUM = "#1f5fa8"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.labelsize": 9.5,
                     "figure.dpi": 150})

T = 100
X = np.arange(-T, T + 1)


def classical_dt(T):
    p = np.zeros(2 * T + 1)
    p[T] = 1.0
    for _ in range(T):
        new = np.zeros_like(p)
        new[1:] += 0.5 * p[:-1]
        new[:-1] += 0.5 * p[1:]
        p = new
    return p


def hadamard_dt(T, coin0=(1 / np.sqrt(2), 1j / np.sqrt(2))):
    """U = S (H x I): coin |0> moves right, |1> moves left."""
    psi = np.zeros((2, 2 * T + 1), dtype=complex)
    psi[0, T], psi[1, T] = coin0
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for _ in range(T):
        c = h @ psi
        new = np.zeros_like(psi)
        new[0, 1:] = c[0, :-1]
        new[1, :-1] = c[1, 1:]
        psi = new
    return (np.abs(psi) ** 2).sum(0)


pc, ph = classical_dt(T), hadamard_dt(T)
even = X % 2 == 0
sd = lambda p: np.sqrt((X ** 2 * p).sum())

fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3), sharex=True, sharey=True)
for ax, p, col, title, lx in [(a, pc, CLASSICAL, "Classical random walk", 0.8),
                              (b, ph, QUANTUM, "Hadamard quantum walk", 0.5)]:
    ax.fill_between(X[even], p[even], color=col, alpha=0.25, lw=0)
    ax.plot(X[even], p[even], color=col, lw=1.0)
    s = sd(p)
    for sgn in (-1, 1):
        ax.axvline(sgn * s, color="#444444", lw=0.7, ls="--")
    ax.text(lx, 0.97, rf"$\sigma \approx {s:.0f}$", transform=ax.transAxes,
            ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_title(title, fontsize=10, color=col)
    ax.set_xlabel("position $x$")
    ax.set_xlim(-T, T)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
a.set_ylabel(rf"$p(x)$ at $t={T}$")
fig.tight_layout()
fig.savefig("fig_walk_side.pdf")
fig.savefig("fig_walk_side.png", dpi=200)
print(f"sd classical {sd(pc):.2f}, sd Hadamard {sd(ph):.2f}, sum {pc.sum():.6f} {ph.sum():.6f}")
