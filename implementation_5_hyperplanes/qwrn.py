"""Quantum-walk re-uploading network (QWRN).

Register: a single walker on the cycle C_n (log2 n qubits), starting at vertex 0.
Layer l (l = 1..L):
    |psi> <- exp(-i V_l) exp(-i t_l(x) A) |psi>,     t_l(x) = theta_l . x + c_l
    A    = adjacency of C_n (the CTQW Hamiltonian), evolved for a data-dependent time
    V_l  = diag(phi_l), a trainable on-site potential
Readout: f(x) = sum_v w_v |<v|psi>|^2 + b  (a trainable diagonal observable), label sign f.

Trainable: theta (L x d), c (L), phi (L x n), w (n), b.  Loss: logistic.
exp(-itA) is evaluated through A's eigendecomposition (Fourier modes, eigenvalues
2cos(2 pi k/n)); on hardware it is QFT . diag(e^{-it lambda}) . QFT^dag.
scikit-learn / qml-benchmarks interface.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin

jax.config.update("jax_enable_x64", True)


def cycle_spectrum(n):
    k = np.arange(n)
    lam = 2 * np.cos(2 * np.pi * k / n)
    F = np.exp(2j * np.pi * np.outer(np.arange(n), k) / n) / np.sqrt(n)  # columns = eigenvectors
    return jnp.asarray(lam), jnp.asarray(F)


class QuantumWalkReuploadingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_vertices=8, n_layers=4, learning_rate=0.01, max_steps=3000,
                 weight_decay=1e-3, init_scale=1.0, random_state=42):
        self.n_vertices = n_vertices
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.init_scale = init_scale
        self.random_state = random_state

    # ------------------------------------------------------------------ model
    def _build(self, d):
        n, L = self.n_vertices, self.n_layers
        lam, F = cycle_spectrum(n)
        Fd = F.conj().T

        def state(params, x):
            psi = jnp.zeros(n, dtype=jnp.complex128).at[0].set(1.0)
            t = params["theta"] @ x + params["c"]                      # (L,)
            for l in range(L):
                psi = F @ (jnp.exp(-1j * t[l] * lam) * (Fd @ psi))    # CTQW for time t_l(x)
                psi = jnp.exp(-1j * params["phi"][l]) * psi            # on-site potential
            return psi

        def f(params, x):
            p = jnp.abs(state(params, x)) ** 2
            return p @ params["w"] + params["b"]

        self._f = jax.jit(jax.vmap(f, in_axes=(None, 0)))
        self._state = jax.jit(jax.vmap(state, in_axes=(None, 0)))

    def _init_params(self, d):
        rng = np.random.default_rng(self.random_state)
        n, L = self.n_vertices, self.n_layers
        return {
            "theta": jnp.asarray(rng.normal(0, self.init_scale / np.sqrt(d), (L, d))),
            "c": jnp.asarray(rng.uniform(0, 2 * np.pi, L)),
            "phi": jnp.asarray(rng.uniform(0, 2 * np.pi, (L, n))),
            "w": jnp.asarray(rng.normal(0, 1.0, n)),
            "b": jnp.asarray(0.0),
        }

    # ------------------------------------------------------------ sklearn API
    def fit(self, X, y):
        X = jnp.asarray(np.asarray(X, float))
        self.classes_ = np.unique(y)
        yy = jnp.asarray(np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0))
        d = X.shape[1]
        self._build(d)
        params = self._init_params(d)
        opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        state = opt.init(params)
        fvec = self._f

        def loss(p):
            return jnp.mean(jax.nn.softplus(-yy * fvec(p, X)))   # logistic loss

        @jax.jit
        def step(p, s):
            l, g = jax.value_and_grad(loss)(p)
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s, l

        for _ in range(self.max_steps):
            params, state, l = step(params, state)
        self.params_ = params
        self.final_loss_ = float(l)
        self.n_qubits_ = int(np.ceil(np.log2(self.n_vertices)))
        return self

    def decision_function(self, X):
        return np.asarray(self._f(self.params_, jnp.asarray(np.asarray(X, float))))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - p, p]
