"""
Gradient variance of the walk ansatz against depth (trainability diagnostic).

For each ansatz and depth, parameters are drawn uniformly from [0, 2pi) and the
gradient of the training loss -- the same mean squared error on the same 140
training images that COBYLA minimises in run_qnn.py -- is computed with
respect to randomly chosen parameters. Var[dL/dw] over draws is the quantity
whose exponential decay defines a barren plateau (McClean et al. 2018).

All models start from the same encoded states (two steps of the real walk
feature map on the 5-qubit grid-walk register) and use the same observable
(Z on every qubit), so only the ansatz differs:
  * walk ansatz, real coin    : T steps, 8 angles per step
  * walk ansatz, complex coin : T steps, 8 angles + 8 phases per step
  * hardware-efficient ansatz : L layers of RY, CNOT ladder, RX on 5 qubits

Gradients are central finite differences on the exact state vector. They are
checked against the exact parameter-shift rule on the hardware-efficient
ansatz, whose generators are Pauli/2.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ibm_dataset import load
from run_qnn import WalkRunner, b_apply_1q, b_apply_cnot
from qnn_core import _ry, _rx

DEPTHS = [1, 2, 4, 8, 16, 32]
N_DRAWS = 200
PARAMS_PER_DRAW = 8
H = 1e-4
SEED = 2026


class HEA:
    """L layers of RY on every qubit, CNOT ladder, RX on every qubit."""

    def __init__(self, n, layers, signs):
        self.n, self.layers, self.signs = n, layers, signs
        self.n_weights = 2 * n * layers

    def outputs(self, enc, w):
        s = enc
        n, k = self.n, 0
        for _ in range(self.layers):
            for q in range(n):
                s = b_apply_1q(s, _ry(w[k]), q, n); k += 1
            for q in range(n - 1):
                s = b_apply_cnot(s, q, q + 1, n)
            for q in range(n):
                s = b_apply_1q(s, _rx(w[k]), q, n); k += 1
        return (np.abs(s) ** 2) @ self.signs


def loss(model, enc, y, w):
    return float(np.mean((model.outputs(enc, w) - y) ** 2))


def fd_grad(model, enc, y, w, k):
    e = np.zeros_like(w); e[k] = H
    return (loss(model, enc, y, w + e) - loss(model, enc, y, w - e)) / (2 * H)


def ps_grad(model, enc, y, w, k):
    """Exact dL/dw_k: parameter shift on f (not on L, which is quadratic in f)."""
    e = np.zeros_like(w); e[k] = np.pi / 2
    df = 0.5 * (model.outputs(enc, w + e) - model.outputs(enc, w - e))
    return float(np.mean(2 * (model.outputs(enc, w) - y) * df))


def measure(model, enc, y, rng):
    grads = []
    for _ in range(N_DRAWS):
        w = rng.uniform(0, 2 * np.pi, model.n_weights)
        ks = rng.choice(model.n_weights, min(PARAMS_PER_DRAW, model.n_weights),
                        replace=False)
        grads.append([fd_grad(model, enc, y, w, k) for k in ks])
    g = np.array(grads)                      # (draws, params)
    var = float(g.var(axis=0).mean())       # variance over draws, mean over params
    boot = []
    for _ in range(500):                     # bootstrap over draws
        idx = rng.integers(0, len(g), len(g))
        boot.append(g[idx].var(axis=0).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return var, float(lo), float(hi), float(np.abs(g).mean())


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr, _, _ = load()
    base = WalkRunner(2, 1, False, False, "walk")
    enc_walk = base.encode(Xtr)                          # (140, n_pad, d)
    enc_flat = enc_walk.reshape(len(Xtr), -1).copy()     # same states, flat
    n, signs = base.n, base.signs

    # -- validation: finite differences vs exact parameter shift (HEA) ------
    hea = HEA(n, 4, signs)
    worst = 0.0
    for _ in range(20):
        w = rng.uniform(0, 2 * np.pi, hea.n_weights)
        k = int(rng.integers(hea.n_weights))
        worst = max(worst, abs(fd_grad(hea, enc_flat, ytr, w, k)
                               - ps_grad(hea, enc_flat, ytr, w, k)))
    print(f"finite difference vs parameter shift: max |delta| = {worst:.1e}")
    assert worst < 1e-6

    out = {"validation_max_abs_delta": worst, "n_qubits": n,
           "n_draws": N_DRAWS, "params_per_draw": PARAMS_PER_DRAW,
           "observable": "Z^{otimes 5}", "loss": "training MSE", "rows": []}
    for depth in DEPTHS:
        for name, model, enc in [
            ("walk, real coin", WalkRunner(2, depth, False, False, "walk"), enc_walk),
            ("walk, complex coin", WalkRunner(2, depth, False, True, "walk"), enc_walk),
            ("hardware-efficient", HEA(n, depth, signs), enc_flat),
        ]:
            t0 = time.time()
            var, lo, hi, mabs = measure(model, enc, ytr, rng)
            row = dict(ansatz=name, depth=depth, n_weights=model.n_weights,
                       var=var, ci_lo=lo, ci_hi=hi, mean_abs_grad=mabs,
                       seconds=round(time.time() - t0, 1))
            out["rows"].append(row)
            print(f"{name:20s} depth {depth:2d}  params {model.n_weights:4d}  "
                  f"Var {var:.3e}  [{lo:.2e}, {hi:.2e}]  "
                  f"<|g|> {mabs:.3e}  {row['seconds']}s", flush=True)
    json.dump(out, open("gradient_variance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
