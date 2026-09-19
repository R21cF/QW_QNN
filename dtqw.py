"""
Discrete-time coined quantum walks on graphs, as Qiskit circuits.

The thesis modifies the walk along two axes and this module is where both are
made concrete:

  axis 1  the coin is a design space, not a fixed choice.  A coin may depend on
          the local structure at a vertex (its degree, or an arbitrary
          per-vertex feature) and on the step index.
  axis 2  the coin's entries need not be real.  The reference architectures
          restrict themselves to real coins -- Dernbach et al. (2019) to real
          Householder reflections for cheap gradients, Zhang et al. (2022) to
          the real Grover diffusion matrix -- so the real/complex contrast is
          a controlled experiment, not a free parameter.

Conventions
-----------
State space  H = H_position (x) H_coin, dimension N * d padded up to a power of
two, with the coin register as the LOW-order qubits: basis index of |v, c> is
v * d_pad + c.  Qiskit's little-endian convention then puts the coin register
first in the qubit list, which is what ``DTQW.circuit`` assembles.

Shift  The flip-flop shift, S|v, i> = |u, j> where u is v's i-th neighbour and
v is u's j-th neighbour.  S is an involution, so it is a permutation matrix and
manifestly unitary.  On |v, i> with i >= deg(v) -- the padding states outside
the walk subspace -- S acts as the identity, which keeps it a permutation on
the whole padded space.

Nothing here assumes a regular graph.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate

__all__ = [
    "coin_hadamard",
    "coin_grover",
    "coin_fourier",
    "coin_householder",
    "coin_parametric",
    "DTQW",
]


# --------------------------------------------------------------------------
# coin families
# --------------------------------------------------------------------------
# Every coin constructor returns a d x d unitary.  ``d`` is the padded coin
# dimension; ``deg`` is the true degree of the vertex the coin sits on.  A coin
# acts as the identity on the deg..d-1 block so that amplitude never leaks into
# directions the vertex does not have.


def _embed(block: np.ndarray, d: int) -> np.ndarray:
    """Place a deg x deg block in the top-left of a d x d identity."""
    deg = block.shape[0]
    if deg == d:
        return block.astype(complex)
    out = np.eye(d, dtype=complex)
    out[:deg, :deg] = block
    return out


def coin_hadamard(d: int, deg: int | None = None, **_) -> np.ndarray:
    """Hadamard coin.  Defined only for d a power of two; the textbook choice
    on the line and the cycle, where d = 2."""
    k = int(np.log2(d))
    if 2 ** k != d:
        raise ValueError(f"Hadamard coin needs d a power of two, got d={d}")
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    out = np.array([[1.0 + 0j]])
    for _i in range(k):
        out = np.kron(out, h)
    return out


def coin_grover(d: int, deg: int | None = None, **_) -> np.ndarray:
    """Grover diffusion coin, 2|s><s| - I on the deg-dimensional subspace.

    This is the coin Zhang et al. (2022) fix for FQWK, and the one the
    walk-based kernel baseline uses.  It is real, symmetric and, for deg > 2,
    the unique coin invariant under permutation of the incident edges.
    """
    deg = d if deg is None else deg
    s = np.ones((deg, 1)) / np.sqrt(deg)
    return _embed(2 * (s @ s.T) - np.eye(deg), d)


def coin_fourier(d: int, deg: int | None = None, **_) -> np.ndarray:
    """DFT coin on the deg-dimensional subspace.  Complex for deg > 2, and so
    the simplest fixed member of the extended (complex) coin family."""
    deg = d if deg is None else deg
    j = np.arange(deg)
    f = np.exp(2j * np.pi * np.outer(j, j) / deg) / np.sqrt(deg)
    return _embed(f, d)


def coin_householder(d: int, deg: int | None = None, *, w: np.ndarray) -> np.ndarray:
    """Real elementary (Householder) reflection I - 2 w w^T / (w^T w).

    Dernbach et al. (2019) use exactly this family for the learned coins in
    QWNN, on the grounds that the forward pass and the gradient are both cheap
    and no unitary projection is needed.  It is real orthogonal by
    construction, which is the restriction axis 2 lifts.
    """
    deg = d if deg is None else deg
    w = np.asarray(w, dtype=float).reshape(-1)[:deg]
    nrm = float(w @ w)
    if nrm < 1e-12:
        return _embed(np.eye(deg), d)
    return _embed(np.eye(deg) - 2.0 * np.outer(w, w) / nrm, d)


def coin_parametric(
    d: int,
    deg: int | None = None,
    *,
    theta: np.ndarray,
    phase: np.ndarray | None = None,
) -> np.ndarray:
    """A parameterised coin covering the real and complex cases under one
    parameterisation, so the two can be compared at equal parameter count.

    The deg-dimensional block is a product of Givens rotations over every pair
    (p, q), p < q, one angle each -- that sweeps the real orthogonal group.
    ``phase``, when given, multiplies on the right by diag(exp(i phi_k)),
    lifting the result out of O(deg) into U(deg).  Setting ``phase=None``
    recovers the real restriction exactly, so the two conditions differ only in
    whether those phases are free.

    theta   length deg*(deg-1)/2
    phase   length deg, or None
    """
    deg = d if deg is None else deg
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n_ang = deg * (deg - 1) // 2
    if theta.size < n_ang:
        raise ValueError(f"need {n_ang} angles for deg={deg}, got {theta.size}")

    m = np.eye(deg, dtype=complex)
    k = 0
    for p in range(deg):
        for q in range(p + 1, deg):
            c, s = np.cos(theta[k]), np.sin(theta[k])
            g = np.eye(deg, dtype=complex)
            g[p, p], g[p, q] = c, -s
            g[q, p], g[q, q] = s, c
            m = g @ m
            k += 1

    if phase is not None:
        phase = np.asarray(phase, dtype=float).reshape(-1)[:deg]
        m = m @ np.diag(np.exp(1j * phase))

    return _embed(m, d)


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------


class DTQW:
    """A coined discrete-time quantum walk on a fixed graph.

    Parameters
    ----------
    graph
        Any networkx graph.  Nodes are relabelled to 0..N-1 in sorted order.
    coin_dim
        Padded coin dimension.  Defaults to the next power of two at or above
        the maximum degree.
    """

    def __init__(self, graph: nx.Graph, coin_dim: int | None = None):
        self.graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        self.n_nodes = self.graph.number_of_nodes()
        self.degrees = np.array(
            [self.graph.degree(v) for v in range(self.n_nodes)], dtype=int
        )
        self.max_degree = int(self.degrees.max()) if self.n_nodes else 0

        d = coin_dim if coin_dim is not None else max(2, self.max_degree)
        self.coin_dim = 1 << (int(d - 1).bit_length())  # round up to power of 2
        if self.coin_dim < self.max_degree:
            raise ValueError("coin_dim smaller than the maximum degree")

        self.n_coin_qubits = int(np.log2(self.coin_dim))
        self.n_pos_qubits = max(1, int(np.ceil(np.log2(max(self.n_nodes, 2)))))
        self.n_qubits = self.n_pos_qubits + self.n_coin_qubits
        self.dim = 1 << self.n_qubits

        # neighbour lists, sorted, so idx(v, u) is well defined
        self._nbrs = [sorted(self.graph.neighbors(v)) for v in range(self.n_nodes)]
        self._nbr_index = [
            {u: i for i, u in enumerate(self._nbrs[v])} for v in range(self.n_nodes)
        ]
        self._shift_perm = self._build_shift_permutation()

    # -- indexing ----------------------------------------------------------

    def index(self, v: int, c: int) -> int:
        """Basis index of |v, c> in the padded space."""
        return v * self.coin_dim + c

    # -- operators as matrices --------------------------------------------

    def _build_shift_permutation(self) -> np.ndarray:
        """Flip-flop shift as a permutation array: perm[i] = j means S|i> = |j>."""
        perm = np.arange(self.dim, dtype=int)
        for v in range(self.n_nodes):
            for i, u in enumerate(self._nbrs[v]):
                j = self._nbr_index[u][v]
                perm[self.index(v, i)] = self.index(u, j)
        return perm

    def shift_matrix(self) -> np.ndarray:
        s = np.zeros((self.dim, self.dim), dtype=complex)
        s[self._shift_perm, np.arange(self.dim)] = 1.0
        return s

    def coin_matrix(self, coin_fn, step: int = 0, **kwargs) -> np.ndarray:
        """Block-diagonal coin operator.

        ``coin_fn`` is called once per vertex as
        ``coin_fn(d=coin_dim, deg=deg_v, vertex=v, step=step, **kwargs)`` and
        must return a coin_dim x coin_dim unitary.  Passing ``vertex`` and
        ``step`` is what makes axis 1 -- structure- and step-dependence --
        expressible without special-casing anything here.
        """
        c = np.eye(self.dim, dtype=complex)
        for v in range(self.n_nodes):
            block = coin_fn(
                d=self.coin_dim, deg=int(self.degrees[v]), vertex=v, step=step, **kwargs
            )
            lo = v * self.coin_dim
            c[lo : lo + self.coin_dim, lo : lo + self.coin_dim] = block
        return c

    def step_matrix(self, coin_fn, step: int = 0, **kwargs) -> np.ndarray:
        """One step, U = S (I (x) C)."""
        return self.shift_matrix() @ self.coin_matrix(coin_fn, step=step, **kwargs)

    def evolution_matrix(self, coin_fn, steps: int, **kwargs) -> np.ndarray:
        u = np.eye(self.dim, dtype=complex)
        for t in range(steps):
            u = self.step_matrix(coin_fn, step=t, **kwargs) @ u
        return u

    # -- states ------------------------------------------------------------

    def initial_state(self, vertex: int, coin_state: np.ndarray | None = None
                      ) -> np.ndarray:
        """A walker localised at ``vertex``.

        ``coin_state`` defaults to the uniform superposition over that vertex's
        real incident directions, which is the standard unbiased start and the
        one that makes the walk's spreading comparable across vertices of
        different degree.
        """
        psi = np.zeros(self.dim, dtype=complex)
        deg = int(self.degrees[vertex])
        if coin_state is None:
            if deg == 0:
                psi[self.index(vertex, 0)] = 1.0
                return psi
            amp = 1.0 / np.sqrt(deg)
            for i in range(deg):
                psi[self.index(vertex, i)] = amp
        else:
            cs = np.asarray(coin_state, dtype=complex).reshape(-1)
            cs = cs / np.linalg.norm(cs)
            for i, a in enumerate(cs[: self.coin_dim]):
                psi[self.index(vertex, i)] = a
        return psi

    def position_distribution(self, psi: np.ndarray) -> np.ndarray:
        """Marginal over the coin register."""
        p = np.abs(psi) ** 2
        return p.reshape(-1, self.coin_dim)[: self.n_nodes].sum(axis=1)

    def walk(self, coin_fn, steps: int, start: int = 0,
             coin_state: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Evolve and return the final statevector."""
        psi = self.initial_state(start, coin_state)
        for t in range(steps):
            psi = self.step_matrix(coin_fn, step=t, **kwargs) @ psi
        return psi

    # -- circuits ----------------------------------------------------------

    def circuit(self, coin_fn, steps: int, start: int = 0,
                coin_state: np.ndarray | None = None,
                label: str = "DTQW", **kwargs) -> QuantumCircuit:
        """The walk as a Qiskit circuit.

        Each step is synthesised as a dense UnitaryGate.  That is the honest
        general construction for an arbitrary graph -- specialised shallower
        constructions exist for highly symmetric or sparse graphs (Loke & Wang)
        but do not apply here -- and it means transpiled depth reported from
        this circuit is an upper bound, which is how Chapter 5 should present
        it.
        """
        qc_coin = QuantumRegister(self.n_coin_qubits, "c")
        qc_pos = QuantumRegister(self.n_pos_qubits, "p")
        qc = QuantumCircuit(qc_coin, qc_pos, name=label)

        all_qubits = list(qc.qubits)
        qc.initialize(self.initial_state(start, coin_state), all_qubits)
        for t in range(steps):
            u = self.step_matrix(coin_fn, step=t, **kwargs)
            qc.append(UnitaryGate(u, label=f"U{t}"), all_qubits)
        return qc

    def step_gate(self, coin_fn, step: int = 0, **kwargs) -> UnitaryGate:
        """One step as a gate, for embedding a walk inside a larger circuit."""
        return UnitaryGate(self.step_matrix(coin_fn, step=step, **kwargs),
                           label=f"W{step}")


# --------------------------------------------------------------------------
# convenience: coin_fn wrappers with the signature coin_matrix expects
# --------------------------------------------------------------------------


def fixed(coin_builder):
    """Adapt a plain coin constructor to the coin_fn calling convention."""
    def fn(d, deg, vertex=None, step=None, **kw):
        return coin_builder(d, deg, **kw)
    return fn


def _phase_vector(angle: float, deg: int) -> np.ndarray:
    """Phases that are not all equal.

    A coin phase vector with every entry the same is diag(e^{i a}) = e^{i a} I,
    a global phase on that vertex's block and therefore unobservable.  Only
    phase DIFFERENCES within a block change anything, so any generator of
    complex coins must produce a non-constant vector or axis 2 is vacuous by
    construction.  Gauge-fixing the first entry to zero also removes the
    redundant overall phase.
    """
    return angle * np.arange(deg, dtype=float)


def structure_dependent(angle_of_degree, *, complex_phases=False):
    """Axis 1: a coin whose angle is a function of the vertex's degree.

    ``angle_of_degree`` maps an integer degree to a single rotation angle; for
    deg > 2 that angle is repeated across the Givens sweep.  Crude on purpose --
    it is the minimal way to make the coin depend on local structure without
    introducing free parameters that would confound the comparison.
    """
    def fn(d, deg, vertex=None, step=None, **kw):
        a = angle_of_degree(deg)
        n_ang = max(1, deg * (deg - 1) // 2)
        th = np.full(n_ang, a, dtype=float)
        ph = _phase_vector(a, deg) if complex_phases else None
        return coin_parametric(d, deg, theta=th, phase=ph)
    return fn


def step_dependent(angle_of_step, *, complex_phases=False):
    """Axis 1, the other half: a coin that changes with the step index."""
    def fn(d, deg, vertex=None, step=0, **kw):
        a = angle_of_step(step)
        n_ang = max(1, deg * (deg - 1) // 2)
        th = np.full(n_ang, a, dtype=float)
        ph = _phase_vector(a, deg) if complex_phases else None
        return coin_parametric(d, deg, theta=th, phase=ph)
    return fn
