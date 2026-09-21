"""Feature maps under one interface, so the comparison is like for like."""

from __future__ import annotations

import numpy as np

from simplewalk import CycleWalkMap

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _apply_1q(state, U, q, n):
    s = state.reshape(2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ij,ajb->aib", U, s).reshape(-1)


def _phase(state, lam, q, n):
    idx = np.arange(1 << n)
    out = state.copy()
    sel = ((idx >> q) & 1) == 1
    out[sel] *= np.exp(1j * lam)
    return out


def _cx(state, c, t, n):
    idx = np.arange(1 << n)
    sel = ((idx >> c) & 1) == 1
    out = state.copy()
    out[idx[sel]] = state[idx[sel] ^ (1 << t)]
    return out


class ZMap:
    """Qiskit's ZFeatureMap: per rep, H on all, then P(2 x_i) on qubit i."""
    name = "Z feature map"

    def __init__(self, n_features, reps=2, scale=1.0):
        self.n = n_features
        self.n_qubits = n_features
        self.reps = reps
        self.scale = scale

    def state(self, x):
        n = self.n
        s = np.zeros(1 << n, dtype=complex); s[0] = 1.0
        xs = self.scale * np.asarray(x, float)
        for _r in range(self.reps):
            for q in range(n):
                s = _apply_1q(s, _H, q, n)
            for q in range(n):
                s = _phase(s, 2.0 * xs[q], q, n)
        return s

    def states(self, X):
        return np.stack([self.state(x) for x in X])


class ZZMap:
    """Qiskit's ZZFeatureMap with full entanglement."""
    name = "ZZ feature map"

    def __init__(self, n_features, reps=2, scale=1.0):
        self.n = n_features
        self.n_qubits = n_features
        self.reps = reps
        self.scale = scale

    def state(self, x):
        n = self.n
        s = np.zeros(1 << n, dtype=complex); s[0] = 1.0
        xs = self.scale * np.asarray(x, float)
        for _r in range(self.reps):
            for q in range(n):
                s = _apply_1q(s, _H, q, n)
            for q in range(n):
                s = _phase(s, 2.0 * xs[q], q, n)
            for i in range(n):
                for j in range(i + 1, n):
                    ang = 2.0 * (np.pi - xs[i]) * (np.pi - xs[j])
                    s = _cx(s, i, j, n)
                    s = _phase(s, ang, j, n)
                    s = _cx(s, i, j, n)
        return s

    def states(self, X):
        return np.stack([self.state(x) for x in X])


class WalkMap:
    """The coined walk on a cycle. log2(n) + 1 qubits."""
    name = "walk feature map"

    def __init__(self, n_features, layers=2, scale=1.0, coin_angle=None):
        self.inner = CycleWalkMap(n_features, layers=layers,
                                  coin_angle=coin_angle, scale=scale)
        self.n_qubits = self.inner.n_qubits
        self.layers = layers
        self.scale = scale

    def states(self, X):
        return self.inner.states(X)

    def state(self, x):
        return self.inner.state(x)


def make(kind, n_features, scale, reps=2, layers=2):
    if kind == "z":
        return ZMap(n_features, reps=reps, scale=scale)
    if kind == "zz":
        return ZZMap(n_features, reps=reps, scale=scale)
    if kind == "walk":
        return WalkMap(n_features, layers=layers, scale=scale)
    raise ValueError(kind)
