"""A discrete-time coined quantum walk used as a feature map for a quantum kernel.

Built from scratch for this thesis.

What is being studied
---------------------
A discrete-time coined walk has several free choices, and this module exposes them
as variables rather than fixing them on principle:

    where the data enters   the coin angle (this variant) or a phase layer on the
                            position register (`entry="position"`)
    coin bias theta0        the coin the data perturbs; pi/4 is Hadamard
    cycle size 2^n          where the walk wraps
    number of steps T       how many times the walk is driven by the features

The point of the experiment is to measure how these choices affect kernel
performance, not to assert in advance that any of them matters. Entering through
the coin makes the walk's trajectory depend on the data -- two inputs produce
different spreading and different interference -- whereas entering as a position
phase leaves the encoding a layer of commuting single-qubit phases that the walk
then redistributes. Which of the two is better, and by how much, is measured.

Construction
------------
    registers   position : n qubits, read as a site in Z_{2^n}
                coin     : 1 qubit
    initial     |coin = 0> (x) |position = 0>     -- localised, so the walk spreads
    step t      C(theta_t)  then  S
        C(th)   [[ cos th,  sin th],
                 [ sin th, -cos th]]      -- real symmetric coin; th = pi/4 is Hadamard
        S       |0>|p> -> |0>|p-1 mod 2^n>,   |1>|p> -> |1>|p+1 mod 2^n>
    data        theta_t = theta0 + lambda * x_{t mod d}   for t = 0 .. T-1
    output      psi(x) in C^{2 . 2^n}
    kernel      K(x, y) = |<psi(x) | psi(y)>|^2

Hyperparameters, all selected by cross-validation on training data only:
    lam     encoding scale
    T       number of walk steps (T >= d; T/d passes over the features)
    n       position qubits.  2^n is the cycle length, and it sets where the walk
            wraps -- the source of the map's periodic structure.
    theta0  coin bias, fixed at pi/4 (Hadamard) unless swept.

Cost per step: one single-qubit coin gate and one controlled cyclic increment on n
qubits. The whole map is T coin gates and T shifts -- no O(d^2) entangling layer.
"""
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
    """(m, 2*2^n) matrix of walk states, one row per sample.

    Vectorised over samples: the coin is the same 2x2 for a given (t, feature value),
    but the value differs per sample, so the contraction is done per sample-batch with
    einsum rather than looping in Python.
    """
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
