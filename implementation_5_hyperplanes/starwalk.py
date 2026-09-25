"""Star-graph quantum-walk product network (SW-QPN).

Encoding: a walker on the star graph S_d (centre vertex 0, leaves 1..d), amplitude-encoded
    |psi_x> = (1, s x_1, ..., s x_d) / sqrt(1 + s^2 |x|^2)          (ceil(log2(d+1)) qubits)
Walker j (j = 1..H, H independent copies of |psi_x>) carries a trainable weighted-star
walk Hamiltonian
    H_j = u_j0 |0><0| + sum_i u_ji (|0><i| + |i><0|)
whose energy is
    E_j(x) = <psi_x|H_j|psi_x> = (u_j0 + 2 s u_j . x) / (1 + s^2 |x|^2),
an affine function of x up to a positive factor.  Readout
    f(x) = a * prod_j tanh(E_j(x)) + b,     label sign f,
so sign f = parity of the H hyperplanes {u_j0 + 2 s u_j . x = 0}: the model class contains
the hyperplanes-and-parity label function exactly once H >= number of hyperplanes.
E_j is measured as the energy of the walk Hamiltonian (Pauli decomposition, or the
first-order phase of exp(-i H_j t) in a Hadamard test).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.base import BaseEstimator, ClassifierMixin

jax.config.update("jax_enable_x64", True)


class StarWalkProductClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_walkers=4, input_scale=1.0, learning_rate=0.01, max_steps=3000,
                 weight_decay=1e-4, init_scale=1.0, n_restarts=1, random_state=42):
        self.n_walkers = n_walkers
        self.input_scale = input_scale
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.init_scale = init_scale
        self.n_restarts = n_restarts
        self.random_state = random_state

    @staticmethod
    def _energies(p, X, s):
        # <psi_x|H_j|psi_x> for all j, from the explicit state and Hamiltonian
        psi = jnp.concatenate([jnp.ones((X.shape[0], 1)), s * X], axis=1)
        psi = psi / jnp.linalg.norm(psi, axis=1, keepdims=True)
        u0, u = p["u0"], p["u"]                            # (H,), (H, d)
        return u0[None, :] * psi[:, :1] ** 2 + 2 * psi[:, :1] * (psi[:, 1:] @ u.T)

    def _f(self, p, X):
        E = self._energies(p, X, self.input_scale)
        return p["a"] * jnp.prod(jnp.tanh(E), axis=1) + p["b"]

    def _init(self, d, seed):
        rng = np.random.default_rng(seed)
        H = self.n_walkers
        return {"u0": jnp.asarray(rng.normal(0, self.init_scale, H)),
                "u": jnp.asarray(rng.normal(0, self.init_scale, (H, d))),
                "a": jnp.asarray(3.0), "b": jnp.asarray(0.0)}

    def fit(self, X, y):
        X = jnp.asarray(np.asarray(X, float))
        self.classes_ = np.unique(y)
        yy = jnp.asarray(np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0))
        opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)

        def loss(p):
            return jnp.mean(jax.nn.softplus(-yy * self._f(p, X)))

        @jax.jit
        def run(p):
            s = opt.init(p)
            def body(c, _):
                p, s = c
                l, g = jax.value_and_grad(loss)(p)
                u, s = opt.update(g, s, p)
                return (optax.apply_updates(p, u), s), l
            (p, s), ls = jax.lax.scan(body, (p, s), None, length=self.max_steps)
            return p, ls[-1]

        best = None
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, self.n_restarts)
        for sd in seeds:                      # restarts selected by TRAINING loss only
            p, l = run(self._init(X.shape[1], int(sd)))
            if best is None or float(l) < best[1]:
                best = (p, float(l))
        self.params_, self.final_loss_ = best
        self.n_qubits_ = int(np.ceil(np.log2(X.shape[1] + 1)))
        return self

    def decision_function(self, X):
        return np.asarray(self._f(self.params_, jnp.asarray(np.asarray(X, float))))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        q = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - q, q]
