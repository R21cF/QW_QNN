"""Quantum-walk fidelity kernels, K(x, x') = |<psi(x)|psi(x')>|^2, used in an SVM.

'product' : one walker per feature on the cycle C_n, CTQW for time s*x_i from vertex 0:
            |psi(x)> = (x)_i exp(-i s x_i A)|0>;  K = prod_i |G_n(s (x_i - x'_i))|^2,
            G_n(t) = <0|exp(-itA)|0> = (1/n) sum_k exp(-2it cos(2 pi k/n))   (-> J_0(2t) as n -> inf)
'reupload': a single walker on C_n, features re-uploaded as walk times, a fixed chirp
            potential D = diag(exp(i pi v^2 / n)) between them (non-commuting, so the
            state is not a product over features):
            |psi(x)> = prod_i [ D exp(-i s x_i A) ] |0>
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


def _cycle(n):
    k = np.arange(n)
    lam = 2 * np.cos(2 * np.pi * k / n)
    F = np.exp(2j * np.pi * np.outer(np.arange(n), k) / n) / np.sqrt(n)
    return lam, F


def walk_states(X, s, n, mode):
    lam, F = _cycle(n)
    Fd = F.conj().T
    if mode == "reupload":
        D = np.exp(1j * np.pi * np.arange(n) ** 2 / n)
        psi = np.zeros((X.shape[0], n), complex); psi[:, 0] = 1
        for i in range(X.shape[1]):
            psi = (psi @ Fd.T) * np.exp(-1j * s * X[:, i:i + 1] * lam[None, :])
            psi = (psi @ F.T) * D[None, :]
        return psi
    raise ValueError


def walk_kernel(A, B, s, n, mode):
    if mode == "product":
        lam, _ = _cycle(n)
        K = np.ones((A.shape[0], B.shape[0]))
        for i in range(A.shape[1]):
            dt = s * (A[:, i][:, None] - B[:, i][None, :])
            G = np.exp(-1j * dt[..., None] * lam).mean(-1)
            K *= np.abs(G) ** 2
        return K
    PA, PB = walk_states(A, s, n, mode), walk_states(B, s, n, mode)
    return np.abs(PA.conj() @ PB.T) ** 2


class QuantumWalkKernelSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, s=0.5, C=1.0, n_vertices=64, mode="product"):
        self.s = s; self.C = C; self.n_vertices = n_vertices; self.mode = mode

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(
            walk_kernel(self.X_, self.X_, self.s, self.n_vertices, self.mode), y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(walk_kernel(np.asarray(X, float), self.X_, self.s, self.n_vertices, self.mode))

    def predict(self, X):
        return self.svc_.predict(walk_kernel(np.asarray(X, float), self.X_, self.s, self.n_vertices, self.mode))
