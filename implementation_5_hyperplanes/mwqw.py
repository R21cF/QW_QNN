"""Multi-walker quantum-walk network (MW-QWN).

H walkers, each on its own cycle C_n (register of log2 n qubits), all starting at vertex 0.
Walker h, layer l = 1..L:
    |psi_h> <- exp(-i V_hl) exp(-i t_hl(x) A) |psi_h>,   t_hl(x) = theta_hl . x + c_hl
    A = adjacency of C_n (CTQW Hamiltonian), V_hl = diag(phi_hl) trainable potential.
Readout: the product observable O = O_1 (x) ... (x) O_H, O_h = diag(tanh(w_h)) (eigenvalues in [-1,1]),
    f(x) = a * prod_h <psi_h|O_h|psi_h> + b.
Each factor g_h(x) = <O_h> is a trigonometric series in the walk times with frequencies
lambda_k - lambda_k' (differences of cycle eigenvalues 2cos(2 pi k/n)); the product observable
multiplies them, matching a label that is a product of signs (parity of hyperplanes).
With L = 1 each factor depends on x only through one learned projection theta_h . x.
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
    F = np.exp(2j * np.pi * np.outer(np.arange(n), k) / n) / np.sqrt(n)
    return jnp.asarray(lam), jnp.asarray(F)


class MultiWalkerQWClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_walkers=3, n_vertices=16, n_layers=1, learning_rate=0.01,
                 max_steps=2000, weight_decay=1e-3, init_scale=1.0, random_state=42):
        self.n_walkers = n_walkers
        self.n_vertices = n_vertices
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.init_scale = init_scale
        self.random_state = random_state

    def _build(self):
        n, L, H = self.n_vertices, self.n_layers, self.n_walkers
        lam, F = cycle_spectrum(n)
        Fd = F.conj().T

        def walker(theta, c, phi, w, x):
            psi = jnp.zeros(n, dtype=jnp.complex128).at[0].set(1.0)
            t = theta @ x + c
            for l in range(L):
                psi = F @ (jnp.exp(-1j * t[l] * lam) * (Fd @ psi))
                if L > 1:
                    psi = jnp.exp(-1j * phi[l]) * psi
            return (jnp.abs(psi) ** 2) @ jnp.tanh(w)

        def f(p, x):
            g = jax.vmap(walker, in_axes=(0, 0, 0, 0, None))(p["theta"], p["c"], p["phi"], p["w"], x)
            return p["a"] * jnp.prod(g) + p["b"]

        self._f = jax.jit(jax.vmap(f, in_axes=(None, 0)))

    def _init(self, d):
        rng = np.random.default_rng(self.random_state)
        n, L, H = self.n_vertices, self.n_layers, self.n_walkers
        return {
            "theta": jnp.asarray(rng.normal(0, self.init_scale / np.sqrt(d), (H, L, d))),
            "c": jnp.asarray(rng.uniform(0, 2 * np.pi, (H, L))),
            "phi": jnp.asarray(rng.uniform(0, 2 * np.pi, (H, L, n))),
            "w": jnp.asarray(rng.normal(0, 1.0, (H, n))),
            "a": jnp.asarray(2.0),
            "b": jnp.asarray(0.0),
        }

    def fit(self, X, y):
        X = jnp.asarray(np.asarray(X, float))
        self.classes_ = np.unique(y)
        yy = jnp.asarray(np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0))
        self._build()
        p = self._init(X.shape[1])
        opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        s = opt.init(p)
        fv = self._f

        def loss(p):
            return jnp.mean(jax.nn.softplus(-yy * fv(p, X)))

        @jax.jit
        def run(p, s):
            def body(carry, _):
                p, s = carry
                l, g = jax.value_and_grad(loss)(p)
                u, s = opt.update(g, s, p)
                return (optax.apply_updates(p, u), s), l
            (p, s), ls = jax.lax.scan(body, (p, s), None, length=self.max_steps)
            return p, ls
        p, ls = run(p, s)
        self.params_ = p
        self.final_loss_ = float(ls[-1])
        self.n_qubits_ = self.n_walkers * int(np.ceil(np.log2(self.n_vertices)))
        return self

    def decision_function(self, X):
        return np.asarray(self._f(self.params_, jnp.asarray(np.asarray(X, float))))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        q = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - q, q]
