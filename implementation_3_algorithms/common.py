"""Shared pieces for implementation_3_algorithms."""

from __future__ import annotations

import numpy as np
import networkx as nx

# --- plotting style shared with implementation_1 --------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2dd"
BLUE = "#2a78d6"     # walk / quantum
ORANGE = "#eb6834"   # comparison 1
GREEN = "#1baf7a"    # comparison 2
NEUTRAL = "#8a8985"  # classical / reference

RC = {
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
}


def grover_coin(d: int) -> np.ndarray:
    """Grover diffusion coin G = 2|s><s| - I on d coin states."""
    if d == 0:
        return np.zeros((0, 0))
    return 2.0 / d * np.ones((d, d)) - np.eye(d)


def walk_operator(G: nx.Graph, coin_fn=None, marked=(), marked_coin=None):
    """Dense step operator W = S (C_v (+) ...) for the coined walk on G."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    d = max(dict(G.degree()).values())
    dim = n * d
    nbrs = {v: list(G.neighbors(v)) for v in nodes}
    marked = set(marked)

    # coin
    C = np.zeros((dim, dim), dtype=complex)
    for v in nodes:
        deg = len(nbrs[v])
        if v in marked:
            c = marked_coin(deg) if marked_coin else -np.eye(deg)
        else:
            c = coin_fn(v, deg) if coin_fn else grover_coin(deg)
        i = idx[v]
        C[i * d:i * d + deg, i * d:i * d + deg] = c
        for k in range(deg, d):
            C[i * d + k, i * d + k] = 1.0
    # shift (flip-flop)
    S = np.eye(dim, dtype=complex)
    for v in nodes:
        for i, u in enumerate(nbrs[v]):
            j = nbrs[u].index(v)
            a, b = idx[v] * d + i, idx[u] * d + j
            S[a, a] = 0.0
            S[a, b] = 1.0
    return S @ C, nodes, d


def position_marginal(psi: np.ndarray, n: int, d: int) -> np.ndarray:
    p = np.abs(psi.reshape(n, d)) ** 2
    return p.sum(axis=1)


def uniform_arc_state(G: nx.Graph, nodes, d) -> np.ndarray:
    """Uniform superposition over arcs (v, i<deg v), the standard walk start."""
    n = len(nodes)
    psi = np.zeros(n * d, dtype=complex)
    for k, v in enumerate(nodes):
        for i in range(G.degree(v)):
            psi[k * d + i] = 1.0
    return psi / np.linalg.norm(psi)


def check_unitary(U: np.ndarray, tol=1e-10) -> float:
    return float(np.abs(U.conj().T @ U - np.eye(U.shape[0])).max())
