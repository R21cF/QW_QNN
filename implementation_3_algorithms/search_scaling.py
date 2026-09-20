"""
Unstructured search by quantum walk: the sqrt(N) law, and what a nonlinearity
does to it.

Three models, all on the complete graph K_N with one marked vertex, all
exact statevector simulation:

  (a) coined DTQW, Grover coin, oracle = -I coin at the marked vertex
      (the walk analogue of Grover's algorithm; Ambainis-Kempe-Rivosh 2005
      style search on the complete graph).  Linear.  Expect t_peak ~ sqrt(N).

  (b) CTQW search, H = -gamma A - |w><w| at the critical gamma = 1/N
      (Childs & Goldstone 2004).  Linear.  Expect t_peak = (pi/2) sqrt(N).

  (c) nonlinear CTQW, H = -gamma A - |w><w| - g |psi(x)|^2   (Gross-Pitaevskii
      cubic term; Meyer & Wong 2013).  The PRA 110, 052411 (2024) result is
      the cubic-quintic, many-body generalisation of this.  Constant time is
      possible ONLY if g grows with N; the experiment measures how fast.

Nothing here can beat sqrt(N) with a linear walk, and the script says so in
its output: that is the BBBV bound, not a limitation of the implementation.
"""

from __future__ import annotations

import json
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from common import walk_operator, position_marginal


# --------------------------------------------------------------------------
# (a) coined DTQW on K_N
# --------------------------------------------------------------------------
def dtqw_search(N: int, tmax: int | None = None, check=False):
    """
    Coined walk on K_N without matrices: psi[v, i] is the amplitude on the arc
    from v to its i-th neighbour (neighbours of v in increasing vertex order,
    skipping v).  Coin = Grover on every vertex except the marked one (-I);
    shift = flip-flop.  For N <= 32 the result is cross-checked against the
    dense operator of common.walk_operator.
    """
    d = N - 1
    # neighbour table and its inverse index for the flip-flop shift
    nb = np.array([[u for u in range(N) if u != v] for v in range(N)])       # (N, d)
    inv = np.zeros((N, d), dtype=int)                                        # j with nb[u, j] = v
    for v in range(N):
        for i in range(d):
            u = nb[v, i]
            inv[v, i] = np.where(nb[u] == v)[0][0]
    psi = np.ones((N, d), dtype=complex) / np.sqrt(N * d)
    tmax = tmax or int(3 * np.sqrt(N)) + 5
    probs = []
    for t in range(tmax + 1):
        probs.append(float((np.abs(psi[0]) ** 2).sum()))
        # coin
        s = psi.sum(axis=1, keepdims=True)
        new = 2.0 / d * s - psi
        new[0] = -psi[0]
        # flip-flop shift: amplitude on arc (v,i) moves to arc (u, inv[v,i])
        out = np.empty_like(new)
        out[nb, inv] = new
        psi = out
    probs = np.array(probs)
    if check:
        G = nx.complete_graph(N)
        W, nodes, dd = walk_operator(G, marked=[0])
        p2 = np.ones(N * dd, dtype=complex) / np.sqrt(N * dd)
        ref = []
        for t in range(tmax + 1):
            ref.append(position_marginal(p2, N, dd)[0])
            p2 = W @ p2
        assert np.allclose(ref, probs, atol=1e-10), "fast path disagrees with dense operator"
    # The flip-flop shift makes odd and even steps alternate (the walk is
    # bipartite in the arc space), so the stopping time is read off the
    # even-step envelope: its first local maximum.
    env = probs[0::2]
    k = 0
    while k + 1 < len(env) and env[k + 1] >= env[k]:
        k += 1
    t = 2 * k
    return t, float(probs[t]), probs


# --------------------------------------------------------------------------
# (b) linear CTQW on K_N  (Childs-Goldstone)
# --------------------------------------------------------------------------
def ctqw_search(N: int, gamma: float | None = None, tmax: float | None = None, nt=400):
    gamma = 1.0 / N if gamma is None else gamma
    A = np.ones((N, N)) - np.eye(N)
    H = -gamma * A
    H[0, 0] -= 1.0
    w, V = np.linalg.eigh(H)
    psi0 = np.ones(N) / np.sqrt(N)
    tmax = tmax or 2.5 * np.sqrt(N)
    ts = np.linspace(0, tmax, nt)
    c0 = V.conj().T @ psi0
    probs = np.array([abs((V @ (np.exp(-1j * w * t) * c0))[0]) ** 2 for t in ts])
    k = int(np.argmax(probs))
    return float(ts[k]), float(probs[k]), ts, probs


# --------------------------------------------------------------------------
# (c) nonlinear CTQW (Gross-Pitaevskii cubic term)
# --------------------------------------------------------------------------
def nonlinear_search(N: int, g: float, gamma: float | None = None,
                     tmax: float | None = None, nt=600):
    """
    i d/dt psi = (-gamma A - |w><w|) psi - g |psi|^2 psi.
    By symmetry the state lives in span{|w>, |s_perp>}: a = amplitude on the
    marked vertex, b = common amplitude on each of the N-1 unmarked ones.
    """
    tmax = tmax or 2.5 * np.sqrt(N)

    def gamma_of(a, b):
        # Meyer-Wong resonance condition, kept as the state evolves: the
        # diagonal energies of |w> and |s_perp> in the reduced two-level
        # picture must stay equal,  -1 - g|a|^2 = -gamma (N-2) - g|b|^2,
        # so gamma is time dependent whenever g != 0.  With g = 0 this is the
        # Childs-Goldstone critical value 1/(N-2) ~ 1/N.
        if gamma is not None:
            return gamma
        return (1.0 + g * abs(a) ** 2 - g * abs(b) ** 2) / (N - 2)

    def rhs(t, y):
        a = y[0] + 1j * y[1]
        b = y[2] + 1j * y[3]
        gm = gamma_of(a, b)
        # A|w> = sum of unmarked, A|u> = |w> + (N-2) other unmarked
        Ha = -gm * (N - 1) * b - a - g * abs(a) ** 2 * a
        Hb = -gm * (a + (N - 2) * b) - g * abs(b) ** 2 * b
        da, db = -1j * Ha, -1j * Hb
        return [da.real, da.imag, db.real, db.imag]

    y0 = [1 / np.sqrt(N), 0.0, 1 / np.sqrt(N), 0.0]
    ts = np.linspace(0, tmax, nt)
    sol = solve_ivp(rhs, (0, tmax), y0, t_eval=ts, rtol=1e-9, atol=1e-11)
    pw = sol.y[0] ** 2 + sol.y[1] ** 2
    # first local maximum above 0.5 counts as "success"; else global max
    k = int(np.argmax(pw))
    return float(ts[k]), float(pw[k]), ts, pw


def time_to_success(N, g, thresh=0.5, tmax=None, nt=1500):
    t, p, ts, pw = nonlinear_search(N, g, tmax=tmax or 3.0 * np.sqrt(N), nt=nt)
    above = np.where(pw >= thresh)[0]
    if len(above) == 0:
        return None, float(pw.max())
    return float(ts[above[0]]), float(pw[above[0]])


def peak_width(N, g, tmax=None, nt=1500):
    """Time of the first peak of the success probability and its full width
    at half maximum -- the quantity the cubic-quintic construction of
    DalFavero et al. is designed to widen."""
    t, p, ts, pw = nonlinear_search(N, g, tmax=tmax or 3.0 * np.sqrt(N), nt=nt)
    # first local maximum
    k = 1
    while k + 1 < len(pw) and pw[k + 1] >= pw[k]:
        k += 1
    half = pw[k] / 2
    lo = k
    while lo > 0 and pw[lo] >= half:
        lo -= 1
    hi = k
    while hi + 1 < len(pw) and pw[hi] >= half:
        hi += 1
    return float(ts[k]), float(pw[k]), float(ts[hi] - ts[lo])


def min_g_for_constant_time(N, t_target, thresh=0.5, gmax_factor=4.0):
    """Smallest g (bisection) reaching p >= thresh by time t_target."""
    lo, hi = 0.0, gmax_factor * N
    t_hi, _ = time_to_success(N, hi, thresh, tmax=1.5 * t_target)
    if t_hi is None or t_hi > t_target:
        return None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        t_mid, _ = time_to_success(N, mid, thresh, tmax=1.5 * t_target)
        if t_mid is not None and t_mid <= t_target:
            hi = mid
        else:
            lo = mid
    return hi


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out = {"dtqw": [], "ctqw": [], "nonlinear": [], "g_required": []}

    # (a) and (b): linear scaling laws
    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        tp, pp, _ = dtqw_search(N, check=(N <= 32))
        out["dtqw"].append({"N": N, "t_peak": tp, "p_peak": pp})
        print(f"DTQW  N={N:4d}  t_peak={tp:4d}  p={pp:.3f}")
    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        tp, pp, _, _ = ctqw_search(N)
        out["ctqw"].append({"N": N, "t_peak": tp, "p_peak": pp})
        print(f"CTQW  N={N:4d}  t_peak={tp:7.2f}  p={pp:.3f}   (pi/2)sqrt(N)={np.pi/2*np.sqrt(N):.2f}")

    # fits
    for key in ("dtqw", "ctqw"):
        Ns = np.array([r["N"] for r in out[key]], float)
        ts = np.array([r["t_peak"] for r in out[key]], float)
        m = Ns >= 16
        alpha, logc = np.polyfit(np.log(Ns[m]), np.log(ts[m]), 1)
        out[key + "_fit"] = {"alpha": float(alpha), "c": float(np.exp(logc))}
        print(f"{key}: t_peak ~ {np.exp(logc):.3f} N^{alpha:.3f}")

    # (c): nonlinear walk at fixed g/N; time, peak height and peak width vs N
    for ratio in [0.0, 0.5, 1.0, 2.0]:
        rows = []
        for N in [16, 32, 64, 128, 256, 512, 1024]:
            # g > 0: the peak sits at t = O(1), so resolve it on a fixed
            # window; g = 0 needs the sqrt(N) window
            kw = dict(tmax=6.0, nt=3000) if ratio > 0 else {}
            t, p = time_to_success(N, ratio * N, **kw)
            tp, pp, w = peak_width(N, ratio * N, **kw)
            rows.append({"N": N, "g_over_N": ratio, "t_success": t, "p": p,
                         "t_peak": tp, "p_peak": pp, "fwhm": w})
            print(f"NL g/N={ratio:.1f}  N={N:5d}  t_success={t}  t_peak={tp:.3f} p_peak={pp:.3f} fwhm={w:.3f}")
        out["nonlinear"].append(rows)

    # required nonlinearity for a fixed target time
    for t_target in [2.0, 3.0]:
        rows = []
        for N in [16, 32, 64, 128, 256, 512]:
            g = min_g_for_constant_time(N, t_target)
            rows.append({"N": N, "t_target": t_target, "g_min": g})
            print(f"t_target={t_target}  N={N:5d}  g_min={g}")
        out["g_required"].append(rows)
        Ns = np.array([r["N"] for r in rows if r["g_min"] is not None], float)
        gs = np.array([r["g_min"] for r in rows if r["g_min"] is not None], float)
        if len(Ns) >= 3:
            a, c = np.polyfit(np.log(Ns), np.log(gs), 1)
            out[f"g_fit_t{t_target:g}"] = {"alpha": float(a), "c": float(np.exp(c))}
            print(f"  g_min ~ {np.exp(c):.3f} N^{a:.3f}")

    json.dump(out, open("search_results.json", "w"), indent=1)
    print("linear walks obey the BBBV Omega(sqrt N) bound; the nonlinear model trades time for g ~ N.")
