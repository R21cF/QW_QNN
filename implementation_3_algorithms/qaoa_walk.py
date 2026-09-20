"""
Quantum walks as QAOA mixers, on a constrained problem where the choice of
mixer is not cosmetic.

Problem.  Balanced Max-Cut (graph bisection): maximise the cut of a weighted
graph on n vertices subject to |S| = n/2.  Feasible strings have Hamming
weight n/2; they are the vertices of the Johnson graph J(n, n/2), two strings
adjacent iff they differ by one swap.

Mixers compared (all with p layers, the cost layer e^{-i gamma C} identical):

  X        transverse-field mixer e^{-i beta sum_i X_i} on the full 2^n space,
           with a quadratic penalty lam (sum_i z_i)^2 in the cost.  This IS a
           continuous-time walk -- on the hypercube -- which does not respect
           the constraint.
  CTQW-J   e^{-i beta A_J}, the continuous-time walk on the Johnson graph
           (Marsh & Wang 2019; identical to the complete XY mixer).  Stays
           in the feasible subspace by construction.
  DTQW-J   one coined discrete-time walk step on the Johnson graph per
           layer, W(beta) = S (I (x) exp(-i beta G)) with G the Grover coin;
           coin register of dimension n/2 * n/2 traced out at the end.
           Also feasibility preserving.

Initial state: uniform superposition over feasible strings (CTQW-J), uniform
over feasible arcs (DTQW-J), |+>^n (X).  Parameters optimised by COBYLA from
R random starts; the best of R is reported, which is the usual protocol and
favours no mixer over another.

Reported: approximation ratio  E[C(x) 1[x feasible]] / C_opt, and the
probability of a feasible outcome.
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.optimize import minimize

SEED = 11
N_INST = 6
n = 8
k = n // 2
P_MAX = 4
RESTARTS = 12
MAXITER = 400


def instance(rng):
    G = nx.gnp_random_graph(n, 0.6, seed=int(rng.integers(1 << 30)))
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(n, 0.6, seed=int(rng.integers(1 << 30)))
    for u, v in G.edges():
        G[u][v]["w"] = float(rng.integers(1, 6))
    return G


def cut_values(G):
    """C(x) for all 2^n bitstrings, x as integer with bit i = vertex i."""
    xs = np.arange(2 ** n)
    bits = (xs[:, None] >> np.arange(n)) & 1
    C = np.zeros(2 ** n)
    for u, v, d in G.edges(data=True):
        C += d["w"] * (bits[:, u] != bits[:, v])
    return C, bits


# --- feasible subspace and the Johnson graph -----------------------------
feas_states = [sum(1 << i for i in c) for c in combinations(range(n), k)]
feas_states.sort()
F = len(feas_states)
fidx = {s: i for i, s in enumerate(feas_states)}
# Johnson adjacency: swap one 1 with one 0
A_J = np.zeros((F, F))
nbrs = [[] for _ in range(F)]
for i, s in enumerate(feas_states):
    ones = [b for b in range(n) if s >> b & 1]
    zeros = [b for b in range(n) if not s >> b & 1]
    for o in ones:
        for z in zeros:
            t = s ^ (1 << o) ^ (1 << z)
            A_J[i, fidx[t]] = 1.0
            nbrs[i].append(fidx[t])
d_J = k * (n - k)
wJ, VJ = np.linalg.eigh(A_J)

# DTQW on J(n,k): arc space F x d_J, flip-flop shift
inv = np.zeros((F, d_J), dtype=int)
nb = np.array(nbrs)
for i in range(F):
    for a, j in enumerate(nbrs[i]):
        inv[i, a] = nbrs[j].index(i)


def dtqw_step(psi, beta):
    # coin exp(-i beta G), G = 2|s><s| - I, G^2 = I  ->  cos b I - i sin b G
    s = psi.sum(axis=1, keepdims=True)
    Gpsi = 2.0 / d_J * s - psi
    new = np.cos(beta) * psi - 1j * np.sin(beta) * Gpsi
    out = np.empty_like(new)
    out[nb, inv] = new
    return out


# --- the three circuits --------------------------------------------------
def run_x(params, C_pen, p):
    gammas, betas = params[:p], params[p:]
    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * C_pen) * psi
        # e^{-i b X} on every qubit
        t = psi.reshape([2] * n)
        c, s = np.cos(b), -1j * np.sin(b)
        for q in range(n):
            t = np.moveaxis(t, q, 0)
            t = np.stack([c * t[0] + s * t[1], s * t[0] + c * t[1]])
            t = np.moveaxis(t, 0, q)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2


def run_ctqw(params, C_f, p):
    gammas, betas = params[:p], params[p:]
    psi = np.ones(F, dtype=complex) / np.sqrt(F)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * C_f) * psi
        psi = VJ @ (np.exp(-1j * b * wJ) * (VJ.conj().T @ psi))
    return np.abs(psi) ** 2


def run_dtqw(params, C_f, p):
    gammas, betas = params[:p], params[p:]
    psi = np.ones((F, d_J), dtype=complex) / np.sqrt(F * d_J)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * g * C_f)[:, None] * psi
        psi = dtqw_step(psi, b)
    return (np.abs(psi) ** 2).sum(axis=1)


def optimise(objective, p, rng):
    best = (np.inf, None)
    for _ in range(RESTARTS):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)])
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.5})
        if res.fun < best[0]:
            best = (res.fun, res.x)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = []
    for inst in range(N_INST):
        G = instance(rng)
        C, bits = cut_values(G)
        feas_mask = bits.sum(axis=1) == k
        C_opt = C[feas_mask].max()
        C_f = C[feas_states]
        lam = 1.0 * max(d["w"] for _, _, d in G.edges(data=True))
        C_pen = C - lam * (bits.sum(axis=1) - k) ** 2
        rec = {"instance": inst, "edges": G.number_of_edges(), "C_opt": float(C_opt),
               "p_feasible_random": float(feas_mask.mean()),
               "ratio_uniform_feasible": float(C_f.mean() / C_opt), "results": {}}
        for p in range(1, P_MAX + 1):
            row = {}
            # X mixer with penalty
            obj = lambda th: -float((run_x(th, C_pen, p) * C_pen).sum())
            _, th = optimise(obj, p, rng)
            pr = run_x(th, C_pen, p)
            row["X"] = {"ratio": float((pr * C * feas_mask).sum() / C_opt),
                        "p_feasible": float(pr[feas_mask].sum()),
                        "ratio_feasible_only": float((pr * C * feas_mask).sum() / pr[feas_mask].sum() / C_opt)}
            # CTQW on Johnson graph
            obj = lambda th: -float((run_ctqw(th, C_f, p) * C_f).sum())
            _, th = optimise(obj, p, rng)
            pr = run_ctqw(th, C_f, p)
            row["CTQW-J"] = {"ratio": float((pr * C_f).sum() / C_opt), "p_feasible": 1.0}
            # DTQW on Johnson graph
            obj = lambda th: -float((run_dtqw(th, C_f, p) * C_f).sum())
            _, th = optimise(obj, p, rng)
            pr = run_dtqw(th, C_f, p)
            row["DTQW-J"] = {"ratio": float((pr * C_f).sum() / C_opt), "p_feasible": 1.0,
                             "p_opt": float(pr[C_f == C_opt].sum())}
            rec["results"][p] = row
            print(f"inst {inst} p={p}: X ratio={row['X']['ratio']:.3f} (feas {row['X']['p_feasible']:.2f}, "
                  f"cond {row['X']['ratio_feasible_only']:.3f})  CTQW-J={row['CTQW-J']['ratio']:.3f}  "
                  f"DTQW-J={row['DTQW-J']['ratio']:.3f}")
        out.append(rec)
    json.dump(out, open("qaoa_results.json", "w"), indent=1)
    # summary
    for p in range(1, P_MAX + 1):
        for m in ["X", "CTQW-J", "DTQW-J"]:
            v = [r["results"][p][m]["ratio"] for r in out]
            print(f"p={p} {m:7s} ratio {np.mean(v):.3f} +- {np.std(v):.3f}")
