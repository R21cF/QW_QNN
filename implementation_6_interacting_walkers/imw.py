"""Interacting multi-walker quantum-walk network (IMW-QNN).

W distinguishable walkers on the same cycle C_n; register (C^n)^{(x)W}, W*log2(n) qubits.
Walker a starts at vertex 0. Layer l = 1..L:
  1. independent CTQWs:  prod_a exp(-i t_al(x) A_a),  t_al(x) = theta_al . x + c_al
     (walker a's walk time is its own learned projection of the input)
  2. on-site interaction: exp(-i g_l sum_{a<b} delta(pos_a, pos_b))   (trainable g_l)
     -- the only gate that couples walkers; it entangles them, and with them the
        feature projections they carry (cross-feature entanglement).
Readout (trainable diagonal observable):
  f(x) = alpha * < prod_a O_a > + sum_a beta_a < O_a > + b,   O_a = diag(tanh(w_a)) on walker a.
With interaction=False the g_l are fixed to 0: the state is a product over walkers
(no cross-feature entanglement) and <prod_a O_a> = prod_a <O_a>.
Multi-particle interacting walks are universal for quantum computation
(Childs, Gosset & Webb, Science 339, 791 (2013)); this is a small trainable instance.
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


class InteractingWalkersClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_walkers=3, n_vertices=8, n_layers=2, interaction=True,
                 learning_rate=0.01, max_steps=1500, weight_decay=1e-2, random_state=42):
        self.n_walkers = n_walkers
        self.n_vertices = n_vertices
        self.n_layers = n_layers
        self.interaction = interaction
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.random_state = random_state

    # ----------------------------------------------------------------- model
    def _build(self):
        W, n, L = self.n_walkers, self.n_vertices, self.n_layers
        lam, F = cycle_spectrum(n)
        Fd = F.conj().T
        # number of coincident pairs sum_{a<b} delta(pos_a, pos_b) on the joint basis
        grids = np.meshgrid(*[np.arange(n)] * W, indexing="ij")
        pairs = sum((grids[a] == grids[b]).astype(float) for a in range(W) for b in range(a + 1, W))
        pairs = jnp.asarray(pairs)
        inter = self.interaction

        def apply_walker(psi, U, a):
            psi = jnp.moveaxis(psi, a, 0)
            psi = jnp.tensordot(U, psi, axes=(1, 0))
            return jnp.moveaxis(psi, 0, a)

        def state(p, x):
            psi = jnp.zeros((n,) * W, dtype=jnp.complex128).at[(0,) * W].set(1.0)
            t = jnp.einsum("ald,d->al", p["theta"], x) + p["c"]          # (W, L)
            for l in range(L):
                for a in range(W):
                    U = F @ (jnp.exp(-1j * t[a, l] * lam)[:, None] * Fd)
                    psi = apply_walker(psi, U, a)
                if inter:
                    psi = psi * jnp.exp(-1j * p["g"][l] * pairs)
            return psi

        def f(p, x):
            prob = jnp.abs(state(p, x)) ** 2
            o = jnp.tanh(p["w"])                                          # (W, n)
            prod = prob
            for a in range(W):   # contract each axis with O_a -> <prod_a O_a>
                prod = jnp.tensordot(o[a], prod, axes=(0, 0))
            singles = []
            for a in range(W):
                marg = prob.sum(axis=tuple(b for b in range(W) if b != a))
                singles.append(marg @ o[a])
            return p["alpha"] * prod + jnp.stack(singles) @ p["beta"] + p["b"]

        self._f = jax.jit(jax.vmap(f, in_axes=(None, 0)))
        self._state = jax.jit(state)

    def _init(self, d):
        rng = np.random.default_rng(self.random_state)
        W, n, L = self.n_walkers, self.n_vertices, self.n_layers
        return {
            "theta": jnp.asarray(rng.normal(0, 1 / np.sqrt(d), (W, L, d))),
            "c": jnp.asarray(rng.uniform(0, 2 * np.pi, (W, L))),
            "g": jnp.asarray(rng.uniform(0, 2 * np.pi, L)),
            "w": jnp.asarray(rng.normal(0, 1.0, (W, n))),
            "alpha": jnp.asarray(2.0),
            "beta": jnp.asarray(rng.normal(0, 0.5, W)),
            "b": jnp.asarray(0.0),
        }

    # ------------------------------------------------------------ sklearn API
    def fit(self, X, y):
        X = jnp.asarray(np.asarray(X, float))
        self.classes_ = np.unique(y)
        yy = jnp.asarray(np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0))
        self._build()
        p = self._init(X.shape[1])
        opt = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        fv = self._f

        def loss(p):
            return jnp.mean(jax.nn.softplus(-yy * fv(p, X)))

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

        self.params_, l = run(p)
        self.final_loss_ = float(l)
        self.n_qubits_ = self.n_walkers * int(np.ceil(np.log2(self.n_vertices)))
        return self

    def decision_function(self, X):
        return np.asarray(self._f(self.params_, jnp.asarray(np.asarray(X, float))))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        q = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - q, q]

    # ------------------------------------------------------------ diagnostics
    def walker_entanglement(self, X):
        """Entanglement entropy (bits) between walker 0 and the other walkers, per input."""
        n, W = self.n_vertices, self.n_walkers
        out = []
        for x in np.asarray(X, float):
            psi = np.asarray(self._state(self.params_, jnp.asarray(x))).reshape(n, -1)
            s = np.linalg.svd(psi, compute_uv=False) ** 2
            s = s[s > 1e-14]
            out.append(float(-(s * np.log2(s)).sum()))
        return np.array(out)
