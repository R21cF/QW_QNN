"""
Sample paths of the symmetric random walk on the line.

Produces, for the introduction of the pre-defence presentation,
  fig_crw_paths.pdf / .png   a handful of sample paths X_t, t = 0..T, from the
      origin, with the diffusive envelope +-sqrt(t) and +-2 sqrt(t).

The paths are sampled (fixed seed), unlike the exact distributions of
walk_vs_crw.py and walk_side_by_side.py; they only illustrate the process.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSICAL = "#c0504d"
LIMIT = "#444444"
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.labelsize": 9.5,
                     "figure.dpi": 150})

T, PATHS = 200, 6
rng = np.random.default_rng(7)
steps = rng.choice([-1, 1], size=(PATHS, T))
X = np.hstack([np.zeros((PATHS, 1), dtype=int), np.cumsum(steps, axis=1)])
t = np.arange(T + 1)

fig, ax = plt.subplots(figsize=(4.2, 2.9))
shades = plt.cm.Reds(np.linspace(0.45, 0.9, PATHS))
for x, c in zip(X, shades):
    ax.plot(t, x, color=c, lw=0.9)
ax.fill_between(t, -2 * np.sqrt(t), 2 * np.sqrt(t), color=CLASSICAL, alpha=0.08, lw=0)
for k, ls in [(1, "--"), (2, ":")]:
    ax.plot(t, k * np.sqrt(t), color=LIMIT, lw=0.8, ls=ls)
    ax.plot(t, -k * np.sqrt(t), color=LIMIT, lw=0.8, ls=ls)
ax.text(T, np.sqrt(T), r" $\sqrt{t}$", va="center", ha="left")
ax.text(T, 2 * np.sqrt(T), r" $2\sqrt{t}$", va="center", ha="left")
ax.axhline(0, color=LIMIT, lw=0.5)
ax.set_xlim(0, T)
ax.set_xlabel("step $t$")
ax.set_ylabel("position $X_t$")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_crw_paths.pdf")
fig.savefig("fig_crw_paths.png", dpi=200)
print("final positions:", X[:, -1], " sqrt(T) =", round(np.sqrt(T), 1))
