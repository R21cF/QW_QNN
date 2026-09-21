"""A discrete-time coined quantum walk used as a feature map for a quantum kernel."""
from __future__ import annotations

import numpy as np

__all__ = ["walk_states", "walk_kernel", "WalkMap"]


def _coin(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [s, -c]], dtype=np.complex128)


def walk_state(x, lam, T, n, theta0=np.pi / 4):
    """Amplitudes of the walk driven by one sample. Returns a flat (2 * 2^n,) vector."""
    N = 1 << n
    psi = np.zeros((2, N), dtype=np.complex128)
    psi[0, 0] = 1.0                       # |coin 0> |position 0>
    d = len(x)
    for t in range(T):
        C = _coin(theta0 + lam * x[t % d])
        psi = C @ psi                     # coin acts on the coin index
        psi = np.stack([np.roll(psi[0], -1),   # coin 0 steps left
                        np.roll(psi[1], +1)])  # coin 1 steps right
    return psi.reshape(-1)


def walk_states(X, lam, T, n, theta0=np.pi / 4, entry="coin"):
    """(m, 2*2^n) matrix of walk states, one row per sample."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m, d = X.shape
    N = 1 << n
    psi = np.zeros((m, 2, N), dtype=np.complex128)
    if entry == "coin":
        psi[:, 0, 0] = 1.0                       # localised; the walk must spread
    else:
        psi[:, :, :] = 1.0 / np.sqrt(2 * N)      # uniform; a phase layer needs support
    sites = np.arange(N)
    for t in range(T):
        if entry == "coin":
            th = theta0 + lam * X[:, t % d]                    # (m,)
        else:
            th = np.full(m, theta0)
            ph = np.exp(1j * lam * np.outer(X[:, t % d], sites))   # (m, N)
            psi = psi * ph[:, None, :]
        c, s_ = np.cos(th), np.sin(th)
        a, b = psi[:, 0, :], psi[:, 1, :]
        psi = np.stack([c[:, None] * a + s_[:, None] * b,
                        s_[:, None] * a - c[:, None] * b], axis=1)
        psi = np.stack([np.roll(psi[:, 0, :], -1, axis=1),
                        np.roll(psi[:, 1, :], +1, axis=1)], axis=1)
    return psi.reshape(m, 2 * N)


def walk_kernel(XA, XB=None, lam=1.0, T=8, n=4, theta0=np.pi / 4, entry="coin"):
    """Fidelity kernel K(x, y) = |<psi(x)|psi(y)>|^2."""
    A = walk_states(XA, lam, T, n, theta0, entry)
    B = A if XB is None else walk_states(XB, lam, T, n, theta0, entry)
    return np.abs(A.conj() @ B.T) ** 2


class WalkMap:
    """Thin holder so the kernel can be passed around with its hyperparameters."""

    def __init__(self, lam=1.0, T=8, n=4, theta0=np.pi / 4):
        self.lam, self.T, self.n, self.theta0 = lam, T, n, theta0

    def states(self, X):
        return walk_states(X, self.lam, self.T, self.n, self.theta0)

    def gram(self, XA, XB=None):
        return walk_kernel(XA, XB, self.lam, self.T, self.n, self.theta0)

    @property
    def qubits(self):
        return self.n + 1

    def __repr__(self):
        return (f"WalkMap(lam={self.lam:g}, T={self.T}, n={self.n}, "
                f"qubits={self.qubits})")
