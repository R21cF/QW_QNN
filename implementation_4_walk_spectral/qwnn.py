"""Quantum-walk spectral classifier (QW-SC) for images on an L x L torus.

Model
-----
  |psi_x>  = x / ||x||                      amplitude encoding, 2*log2(L) qubits
                                            (row register (x) column register)
  A_r, A_c = adjacency of the cycle C_L acting on the row / column register;
             H = A_r (x) I + I (x) A_c is the continuous-time quantum walk (CTQW)
             Hamiltonian of the L x L torus, the image's pixel-adjacency graph.
  O_w      = sum_{a,b} w_ab  Pi_a(A_r) (x) Pi_b(A_c)
             Pi_a(A) = spectral projector of A onto eigenvalue 2 cos(2 pi a / L),
             a = 0 .. L/2.   O_w is an arbitrary real function of the two commuting
             walk Hamiltonians, i.e. a trainable observable in the walk's eigenbasis.
  f_w(x)   = <psi_x| O_w |psi_x> + b,      label = sign f_w(x)

Circuit: the unitary diagonalising the cycle CTQW is the QFT, so measuring O_w is
  StatePreparation(x) -> QFT on row register, QFT on column register -> measure all
  qubits in the computational basis -> accumulate w_{bin(k_r), bin(k_c)}.

The same QFTs are how the CTQW e^{-iHt} itself is compiled on a torus
(QFT^dag . diag . QFT); O_w commutes with e^{-iHt} for every t, so f_w is a
conserved quantity of the walk: it reads the input state's walk-energy
distribution.

The interface follows qml-benchmarks (scikit-learn BaseEstimator /
ClassifierMixin with fit, predict, predict_proba).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def cycle_adjacency(L):
    A = np.zeros((L, L))
    for i in range(L):
        A[i, (i + 1) % L] = A[(i + 1) % L, i] = 1.0
    return A


def momentum_bin(L):
    """k -> a = min(k, L-k): momenta k and L-k share the eigenvalue 2cos(2 pi k/L)."""
    k = np.arange(L)
    return np.minimum(k, L - k)


def walk_spectral_probs(X, L):
    """Exact outcome distribution of the circuit, P(k_r, k_c) = |<k_r k_c|QFT(x)QFT|psi_x>|^2,
    shape (n, L, L).  Computed with the unitary 2D DFT (sign of the exponent does not
    matter for probabilities)."""
    imgs = X.reshape(-1, L, L)
    F = np.fft.fft2(imgs, norm="ortho")
    P = np.abs(F) ** 2
    P /= P.sum(axis=(1, 2), keepdims=True)
    return P


def bin_probs(P, L):
    """Probability of each joint spectral projector Pi_a (x) Pi_b: shape (n, (L/2+1)^2)."""
    a = momentum_bin(L)
    nb = L // 2 + 1
    out = np.zeros((P.shape[0], nb, nb))
    for i in range(L):
        for j in range(L):
            out[:, a[i], a[j]] += P[:, i, j]
    return out.reshape(P.shape[0], -1)


def sample_probs(P, shots, rng):
    """Finite-shot estimate of P from `shots` computational-basis measurements."""
    n, L, _ = P.shape
    flat = P.reshape(n, -1)
    est = np.stack([rng.multinomial(shots, p / p.sum()) for p in flat]) / shots
    return est.reshape(n, L, L)


class QuantumWalkSpectralClassifier(BaseEstimator, ClassifierMixin):
    """Trainable-observable quantum-walk classifier.

    Args:
        C (float): inverse L2 regularisation strength on the observable weights w.
        shots (int or None): measurement shots per input for features at predict and
            fit time; None = exact expectation values (as the suite's default.qubit runs).
        random_state (int): seed for shot sampling.
    """

    def __init__(self, C=1.0, shots=None, random_state=42):
        self.C = C
        self.shots = shots
        self.random_state = random_state

    # --- quantum part -------------------------------------------------------
    def _features(self, X):
        L = int(round(np.sqrt(X.shape[1])))
        P = walk_spectral_probs(X, L)
        if self.shots is not None:
            P = sample_probs(P, self.shots, self.rng_)
        return bin_probs(P, L)

    # --- scikit-learn API ---------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.rng_ = np.random.default_rng(self.random_state)
        self.L_ = int(round(np.sqrt(X.shape[1])))
        assert self.L_ ** 2 == X.shape[1] and self.L_ & (self.L_ - 1) == 0
        self.n_qubits_ = 2 * int(np.log2(self.L_))
        self.classes_ = np.unique(y)
        Phi = self._features(X)
        # f_w = <O_w> + b is linear in w, so the logistic loss is convex in (w, b)
        self.lr_ = LogisticRegression(C=self.C, max_iter=10000)
        self.lr_.fit(Phi, y)
        nb = self.L_ // 2 + 1
        w = self.lr_.coef_[0]
        self.params_ = {"w": w.reshape(nb, nb) if w.size == nb * nb else w,
                        "b": float(self.lr_.intercept_[0])}
        return self

    def decision_function(self, X):
        return self.lr_.decision_function(self._features(np.asarray(X, float)))

    def predict(self, X):
        return self.lr_.predict(self._features(np.asarray(X, float)))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._features(np.asarray(X, float)))
