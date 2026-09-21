"""
Fast state-space evolver for the uniform-coin case.

dtqw.DTQW builds dense (N*d) x (N*d) operators, which is the right thing for
circuit synthesis and for reporting resource costs but far too slow inside an
optimiser loop.  When the coin is the same matrix at every vertex, one step is

    psi <- S (psi C^T)

with psi held as an (N_pad, d) array: the coin is a single d x d matmul
broadcast over vertices, O(N d^2), and the shift is an index permutation.  That
is the same operator dtqw.DTQW.step_matrix builds -- ``check_against_dtqw``
below asserts it.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from dtqw import DTQW


class FastUniformWalk:
    def __init__(self, graph: nx.Graph):
        self.ref = DTQW(graph)
        self.n_nodes = self.ref.n_nodes
        self.d = self.ref.coin_dim
        self.n_pad = self.ref.dim // self.d
        self.degrees = self.ref.degrees
        # flat permutation -> (vertex, coin) index pair form
        self.perm = self.ref._shift_perm

    def initial(self, start: int) -> np.ndarray:
        return self.ref.initial_state(start).reshape(self.n_pad, self.d)

    def step(self, psi: np.ndarray, coin: np.ndarray) -> np.ndarray:
        return self._shift(psi @ coin.T)

    def _shift(self, psi: np.ndarray) -> np.ndarray:
        flat = psi.reshape(-1)
        out = np.zeros_like(flat)
        out[self.perm] = flat
        return out.reshape(self.n_pad, self.d)

    def evolve(self, coin: np.ndarray, steps: int, start: int) -> np.ndarray:
        psi = self.initial(start)
        for _t in range(steps):
            psi = self._shift(psi @ coin.T)
        return psi

    def distribution(self, coin: np.ndarray, steps: int, start: int) -> np.ndarray:
        psi = self.evolve(coin, steps, start)
        return (np.abs(psi) ** 2).sum(axis=1)[: self.n_nodes]


def check_against_dtqw(graph, steps=5, start=0, seed=0):
    """The fast path must agree with the dense operator it replaces."""
    from scipy.stats import unitary_group

    w = DTQW(graph)
    f = FastUniformWalk(graph)
    c = unitary_group.rvs(w.coin_dim, random_state=seed)

    def coin_fn(d, deg, vertex=None, step=None, **kw):
        return c

    slow = w.walk(coin_fn, steps=steps, start=start)
    fast = f.evolve(c, steps, start).reshape(-1)
    return float(np.abs(slow - fast).max())


if __name__ == "__main__":
    for name, g in [("cycle-16", nx.cycle_graph(16)),
                    ("petersen", nx.petersen_graph()),
                    ("Q4", nx.hypercube_graph(4)),
                    ("4-reg-16", nx.random_regular_graph(4, 16, seed=3))]:
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        err = check_against_dtqw(g)
        print(f"{name:<12} max |fast - dense| = {err:.2e}  "
              f"{'OK' if err < 1e-12 else 'MISMATCH'}")
