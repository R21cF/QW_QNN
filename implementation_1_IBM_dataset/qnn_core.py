"""Statevector cores for the QNN comparison."""

from __future__ import annotations

import numpy as np

from ibm_dataset import N_PIXELS, grid_edges

# --------------------------------------------------------------------------
# gate-model primitives
# --------------------------------------------------------------------------


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rx(t):
    c, s = np.cos(t / 2), -1j * np.sin(t / 2)
    return np.array([[c, s], [s, c]], dtype=complex)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _p(lam):
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)


def apply_1q(state, U, q, n):
    s = state.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ij,ajb->aib", U, s).reshape(-1)


def apply_cnot(state, c, t, n):
    idx = np.arange(1 << n)
    sel = ((idx >> c) & 1) == 1
    out = state.copy()
    out[idx[sel]] = state[idx[sel] ^ (1 << t)]
    return out


def parity_z(state, n):
    """<Z (x) Z (x) ... (x) Z>: sign is the parity of the basis index."""
    idx = np.arange(1 << n)
    signs = 1.0 - 2.0 * (np.array([bin(i).count("1") for i in idx]) & 1)
    return float(np.real((np.abs(state) ** 2) @ signs))


_PARITY_CACHE: dict[int, np.ndarray] = {}


def parity_signs(n):
    if n not in _PARITY_CACHE:
        idx = np.arange(1 << n)
        _PARITY_CACHE[n] = 1.0 - 2.0 * (
            np.array([bin(i).count("1") for i in idx]) & 1
        )
    return _PARITY_CACHE[n]


# --------------------------------------------------------------------------
# the course's model
# --------------------------------------------------------------------------


def z_feature_map(x, n, reps=2):
    """Qiskit's ZFeatureMap: per rep, H on every qubit then P(2 x_i) on i."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0
    for _r in range(reps):
        for q in range(n):
            state = apply_1q(state, _H, q, n)
        for q in range(n):
            state = apply_1q(state, _p(2.0 * x[q]), q, n)
    return state


def ibm_ansatz(state, weights, n, cnot_pairs):
    """RY layer, CNOT layer, RX layer -- 2n parameters."""
    for q in range(n):
        state = apply_1q(state, _ry(weights[q]), q, n)
    for a, b in cnot_pairs:
        state = apply_cnot(state, a, b, n)
    for q in range(n):
        state = apply_1q(state, _rx(weights[n + q]), q, n)
    return state


class GateQNN:
    """The course's variational classifier."""

    def __init__(self, cnot_pairs, n=N_PIXELS, reps=2):
        self.n = n
        self.reps = reps
        self.cnot_pairs = list(cnot_pairs)
        self.n_weights = 2 * n
        self.signs = parity_signs(n)

    def forward(self, x, w):
        s = ibm_ansatz(z_feature_map(x, self.n, self.reps), w, self.n,
                       self.cnot_pairs)
        return float((np.abs(s) ** 2) @ self.signs)

    def describe(self):
        return dict(qubits=self.n, weights=self.n_weights,
                    cnots=len(self.cnot_pairs))


# --------------------------------------------------------------------------
# walk-model primitives
# --------------------------------------------------------------------------


def givens_chain(deg, theta, phase=None):
    """
    One-parameter real coin: a chain of Givens rotations (0,1), (1,2), ... all
    by the same angle. Degree-agnostic, equals the identity at theta = 0.
    """
    m = np.eye(deg, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    for k in range(deg - 1):
        g = np.eye(deg, dtype=complex)
        g[k, k], g[k, k + 1] = c, -s
        g[k + 1, k], g[k + 1, k + 1] = s, c
        m = g @ m
    if phase is not None:
        m = m @ np.diag(np.exp(1j * phase * np.arange(deg)))
    return m


class GridWalk:
    """DTQW on the pixel-grid graph, with per-vertex coins."""

    def __init__(self, edges=None, n_nodes=N_PIXELS):
        edges = grid_edges() if edges is None else edges
        self.n_nodes = n_nodes
        self.nbrs = [[] for _ in range(n_nodes)]
        for a, b in edges:
            self.nbrs[a].append(b)
            self.nbrs[b].append(a)
        for v in range(n_nodes):
            self.nbrs[v].sort()
        self.degrees = np.array([len(self.nbrs[v]) for v in range(n_nodes)])
        self.max_degree = int(self.degrees.max())

        self.d = 1 << int(self.max_degree - 1).bit_length()
        self.n_coin_q = int(np.log2(self.d))
        self.n_pos_q = max(1, int(np.ceil(np.log2(n_nodes))))
        self.n_qubits = self.n_pos_q + self.n_coin_q
        self.n_pad = 1 << self.n_pos_q
        self.dim = self.n_pad * self.d

        self.nbr_index = [{u: i for i, u in enumerate(self.nbrs[v])}
                          for v in range(n_nodes)]
        self.perm = self._shift_perm()
        self.signs = parity_signs(self.n_qubits)
        self._psi0 = self._uniform_start()

    def _shift_perm(self):
        perm = np.arange(self.dim, dtype=int)
        for v in range(self.n_nodes):
            for i, u in enumerate(self.nbrs[v]):
                perm[v * self.d + i] = u * self.d + self.nbr_index[u][v]
        return perm

    def _uniform_start(self):
        """Uniform over the real (vertex, direction) pairs."""
        psi = np.zeros((self.n_pad, self.d), dtype=complex)
        for v in range(self.n_nodes):
            psi[v, : self.degrees[v]] = 1.0
        return psi / np.linalg.norm(psi)

    def coin_stack(self, thetas, phases=None):
        """(n_pad, d, d) stack: one coin per vertex, identity on padding."""
        C = np.tile(np.eye(self.d, dtype=complex), (self.n_pad, 1, 1))
        for v in range(self.n_nodes):
            deg = int(self.degrees[v])
            ph = None if phases is None else float(phases[v])
            C[v, :deg, :deg] = givens_chain(deg, float(thetas[v]), ph)
        return C

    def step(self, psi, C):
        return self._shift(np.einsum("vij,vj->vi", C, psi))

    def _shift(self, psi):
        flat = psi.reshape(-1)
        out = np.empty_like(flat)
        out[self.perm] = flat
        return out.reshape(self.n_pad, self.d)

    def evolve(self, psi, coin_stacks):
        for C in coin_stacks:
            psi = self._shift(np.einsum("vij,vj->vi", C, psi))
        return psi

    def expectation(self, psi):
        return float((np.abs(psi.reshape(-1)) ** 2) @ self.signs)


class WalkQNN:
    """Walk feature map and/or walk ansatz."""

    def __init__(self, fm_steps=2, ans_steps=2, complex_fm=False,
                 complex_ans=False, ansatz="walk", scale=1.0,
                 fixed_phase=None):
        """
        fixed_phase: if given together with complex_ans, the ansatz coins are
        complex with the phase ramp frozen at this value and only the rotation
        angle trained -- one parameter per vertex per step, as for the real
        coin. This isolates realness from parameter count and depth.
        """
        self.g = GridWalk()
        self.fixed_phase = fixed_phase
        self.fm_steps = fm_steps
        self.ans_steps = ans_steps
        self.complex_fm = complex_fm
        self.complex_ans = complex_ans
        self.ansatz = ansatz
        self.scale = scale
        self.n = self.g.n_qubits

        trains_phase = complex_ans and fixed_phase is None
        per_step = self.g.n_nodes * (2 if trains_phase else 1)
        if ansatz == "walk":
            self.n_weights = ans_steps * per_step
        else:  # rotation ansatz on the walk register
            self.n_weights = 2 * self.g.n_qubits

    # -- feature map -------------------------------------------------------

    def _fm_stacks(self, x):
        stacks = []
        for t in range(self.fm_steps):
            th = self.scale * np.asarray(x, dtype=float)
            ph = th if self.complex_fm else None
            stacks.append(self.g.coin_stack(th, ph))
        return stacks

    # -- ansatz ------------------------------------------------------------

    def _ans_stacks(self, w):
        nn = self.g.n_nodes
        stacks, k = [], 0
        for _t in range(self.ans_steps):
            th = w[k:k + nn]; k += nn
            if self.complex_ans and self.fixed_phase is not None:
                ph = np.full(nn, float(self.fixed_phase))
            elif self.complex_ans:
                ph = w[k:k + nn]; k += nn
            else:
                ph = None
            stacks.append(self.g.coin_stack(th, ph))
        return stacks

    def forward(self, x, w):
        psi = self.g._psi0
        if self.fm_steps > 0:
            psi = self.g.evolve(psi, self._fm_stacks(x))
        if self.ansatz == "walk":
            psi = self.g.evolve(psi, self._ans_stacks(w))
            return self.g.expectation(psi)
        # gate ansatz acting on the walk register
        flat = psi.reshape(-1).copy()
        n = self.n
        for q in range(n):
            flat = apply_1q(flat, _ry(w[q]), q, n)
        for q in range(n - 1):
            flat = apply_cnot(flat, q, q + 1, n)
        for q in range(n):
            flat = apply_1q(flat, _rx(w[n + q]), q, n)
        return float((np.abs(flat) ** 2) @ self.g.signs)

    def describe(self):
        return dict(qubits=self.n, weights=self.n_weights,
                    fm_steps=self.fm_steps, ans_steps=self.ans_steps,
                    complex_fm=self.complex_fm, complex_ans=self.complex_ans,
                    fixed_phase=self.fixed_phase)
