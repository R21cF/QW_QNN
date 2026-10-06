"""
Gradient variance of parametrised continuous-time-walk ansaetze on the cycle C_N, N = 2^h.

Section 5.1.4 of the thesis. The classifier of implementation_7 has no trainable parameter inside the walk;
this script measures what happens to the gradient if the walk is made trainable in the two natural ways:

  (A) walk-time ansatz      U(t) = exp(-i t A),            one parameter, t ~ Uniform[0, 8]
  (B) Fourier-phase ansatz  U(theta) = F^dag diag(e^{-i theta_k}) F,  N parameters, theta_k ~ Uniform[0, 2 pi)
      -- every translation-invariant unitary on the cycle is of this form, so (B) is the most general
      "walk" on C_N; (A) is the one-parameter curve theta_k = 2 t cos(2 pi k / N) inside it.

The input state is a basis state |v>, v uniform over the vertices (the classifier's position register holds a
basis state fixed by the data), and three traceless +-1 diagonal read-outs are used:
  half-cycle   o_u = +1 for u < N/2, -1 otherwise   (Z on the top position qubit; the DLP label observable)
  alternating  o_u = (-1)^u                        (Z on the bottom position qubit)
  parity       o_u = (-1)^{popcount(u)}            (Z^{(x)h}, the global cost of the earlier chapters)

For (B) the variance is known in closed form: Var[d<O>/d theta_j] = 2/N^2 for every traceless +-1 diagonal O
(derived in the text); the script checks it. For (A) the gradient is <psi_t| i[A, O] |psi_t>.

Outputs: fig_gradvar_ctqw.pdf/.png, tab_gradvar_ctqw.tex, gradvar_ctqw.json.
"""
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # outputs are written next to this script
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
H_LIST = list(range(3, 15))
SAMPLES = 4000
T_MAX = 8.0
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 150})


def observables(N):
    u = np.arange(N)
    return {"half-cycle": np.where(u < N // 2, 1.0, -1.0),
            "alternating": (-1.0) ** u,
            "parity": (-1.0) ** np.array([bin(x).count("1") for x in u])}


def lam(N):
    return 2 * np.cos(2 * np.pi * np.arange(N) / N)


def evolve(vec, phases):
    """F^dag diag(e^{-i phases}) F vec, with F the unitary DFT (numpy fft convention)."""
    return np.fft.ifft(np.exp(-1j * phases) * np.fft.fft(vec))


def grad_time(N, obs, v, t):
    """d<O>/dt for U(t) = e^{-itA}: 2 Re <psi| O |dpsi>, dpsi = -i A psi."""
    e = np.zeros(N, complex); e[v] = 1
    psi = evolve(e, t * lam(N))
    Apsi = np.roll(psi, 1) + np.roll(psi, -1)
    return 2 * np.real(np.vdot(psi, obs * (-1j * Apsi)))


def grad_fourier(N, obs, v, theta, j):
    e = np.zeros(N, complex); e[v] = 1
    fe = np.fft.fft(e)
    psi = np.fft.ifft(np.exp(-1j * theta) * fe)
    d = np.zeros(N, complex); d[j] = -1j * np.exp(-1j * theta[j]) * fe[j]
    dpsi = np.fft.ifft(d)
    return 2 * np.real(np.vdot(psi, obs * dpsi))


def bootstrap_ci(g, B=400):
    v = g.var()
    bs = np.array([rng.choice(g, len(g)).var() for _ in range(B)])
    return float(v), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


res = {"A": {}, "B": {}}
for h in H_LIST:
    N = 2 ** h
    obs = observables(N)
    for name, o in obs.items():
        gA = np.array([grad_time(N, o, rng.integers(N), rng.uniform(0, T_MAX)) for _ in range(SAMPLES)])
        gB = np.array([grad_fourier(N, o, rng.integers(N), rng.uniform(0, 2 * np.pi, N), rng.integers(N))
                       for _ in range(SAMPLES)])
        res["A"].setdefault(name, {})[h] = bootstrap_ci(gA)
        res["B"].setdefault(name, {})[h] = bootstrap_ci(gB)
    print(h, {k: round(res["A"][k][h][0], 5) for k in obs}, {k: round(res["B"][k][h][0] * N ** 2, 3) for k in obs}, flush=True)

# slopes: log2 Var per qubit, least squares over h
def slope(d):
    hs = np.array(sorted(d)); ys = np.log2([d[h][0] for h in hs])
    return float(np.polyfit(hs, ys, 1)[0])

summary = {"slopes_A": {k: slope(res["A"][k]) for k in res["A"]},
           "slopes_B": {k: slope(res["B"][k]) for k in res["B"]},
           "B_times_N2": {k: {h: res["B"][k][h][0] * 4 ** h for h in H_LIST} for k in res["B"]},
           "samples": SAMPLES, "t_max": T_MAX}
json.dump({"res": {a: {k: {str(h): v for h, v in d.items()} for k, d in r.items()} for a, r in res.items()},
           "summary": summary}, open("gradvar_ctqw.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# figure
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
colors = {"half-cycle": "#1f5fa8", "alternating": "#c0504d", "parity": "#6a9c3a"}
labels = {"half-cycle": "half-cycle $Z_{\\mathrm{top}}$", "alternating": "alternating $Z_{\\mathrm{bottom}}$",
          "parity": "parity $Z^{\\otimes h}$"}
for a, key, title in zip(ax, ("A", "B"), ("(a) walk-time ansatz $e^{-itA}$, $t\\sim U[0,8]$",
                                         "(b) Fourier-phase ansatz, $N$ parameters")):
    for name in ("half-cycle", "alternating", "parity"):
        hs = np.array(H_LIST)
        v = np.array([res[key][name][h][0] for h in hs])
        lo = np.array([res[key][name][h][1] for h in hs]); hi = np.array([res[key][name][h][2] for h in hs])
        a.errorbar(hs, v, yerr=[v - lo, hi - v], color=colors[name], marker="o", ms=3, lw=1, capsize=2, label=labels[name])
    a.set_yscale("log"); a.set_xlabel("position qubits $h$ ($N = 2^h$)")
    a.set_title(title, loc="left", fontsize=8.5)
ax[1].plot(H_LIST, [2 / 4 ** h for h in H_LIST], color="#444444", ls="--", lw=0.9, label="$2/N^2$ (exact)")
ax[0].plot(H_LIST, [1 / 2 ** h for h in H_LIST], color="#444444", ls=":", lw=0.9, label="$\\propto 1/N$")
ax[0].set_ylabel("Var$[\\partial_\\theta \\langle O\\rangle]$")
ax[0].legend(frameon=False); ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("fig_gradvar_ctqw.pdf"); fig.savefig("fig_gradvar_ctqw.png", dpi=200)

# table
def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

with open("tab_gradvar_ctqw.tex", "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\\small\n")
    f.write("\\caption[Gradient variance of the walk ans\\\"atze]{Variance of the gradient of $\\langle O\\rangle$ with "
            "respect to one parameter, at uniformly random parameters and a uniformly random start vertex, for the "
            "walk-time ansatz (A) and the Fourier-phase ansatz (B) on the cycle $C_{2^h}$, with three traceless "
            "$\\pm1$ diagonal read-outs. $4000$ samples per entry. The last column is the least-squares slope of "
            "$\\log_2\\mathrm{Var}$ per position qubit over $h = 3, \\ldots, 14$; $-2$ is the exact value for (B).}\n")
    f.write("\\label{tab:gradvarctqw}\n\\begin{tabular}{llrrrrr}\n\\hline\n")
    f.write("Ansatz & Read-out & $h=4$ & $h=8$ & $h=12$ & $h=14$ & slope \\\\\n\\hline\n")
    for key, kname in (("A", "(A) walk time $t$"), ("B", "(B) Fourier phases")):
        for name in ("half-cycle", "alternating", "parity"):
            vals = " & ".join(sci(res[key][name][h][0]) for h in (4, 8, 12, 14))
            f.write(f"{kname} & {name} & {vals} & ${summary['slopes_' + key][name]:.2f}$ \\\\\n")
    f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
print(open("tab_gradvar_ctqw.tex").read())
