"""
Train and evaluate every model on the course's line-detection task.

Two speedups, both exact:
  * the feature map does not depend on the weights, so every sample's
    post-encoding state is computed once and reused for the whole optimisation
  * the ansatz is the same operator for every sample, so it is applied to the
    batch of states at once

``--verify`` asserts the batched path matches the per-sample forward in
qnn_core, so the speed changes nothing but the runtime.

Reported per model: test accuracy over several weight initialisations, weight
count, qubit count, and transpiled depth -- accuracy alone would hide that the
walk models cost two orders of magnitude more depth.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

from ibm_dataset import load, horizontal_pairs, N_PIXELS
from qnn_core import (
    GateQNN, WalkQNN, GridWalk, z_feature_map, parity_signs,
    _ry, _rx,
)

MAXITER = 500
N_SEEDS = 5


# ----------------------------------------------------------------- batch --


def b_apply_1q(states, U, q, n):
    s = states.reshape(states.shape[0], 2 ** (n - 1 - q), 2, 2 ** q)
    return np.einsum("ij,bajc->baic", U, s).reshape(states.shape[0], -1)


def b_apply_cnot(states, c, t, n):
    idx = np.arange(1 << n)
    sel = ((idx >> c) & 1) == 1
    out = states.copy()
    out[:, idx[sel]] = states[:, idx[sel] ^ (1 << t)]
    return out


class GateRunner:
    """Course model: precomputed ZFeatureMap states, batched ansatz."""

    name_prefix = "gate"

    def __init__(self, cnot_pairs, reps=2):
        self.n = N_PIXELS
        self.pairs = list(cnot_pairs)
        self.reps = reps
        self.n_weights = 2 * self.n
        self.signs = parity_signs(self.n)

    def encode(self, X):
        return np.stack([z_feature_map(x, self.n, self.reps) for x in X])

    def outputs(self, enc, w):
        s = enc
        for q in range(self.n):
            s = b_apply_1q(s, _ry(w[q]), q, self.n)
        for a, b in self.pairs:
            s = b_apply_cnot(s, a, b, self.n)
        for q in range(self.n):
            s = b_apply_1q(s, _rx(w[self.n + q]), q, self.n)
        return (np.abs(s) ** 2) @ self.signs


class WalkRunner:
    """Walk feature map, then either a walk ansatz or a rotation ansatz."""

    def __init__(self, fm_steps=2, ans_steps=2, complex_fm=False,
                 complex_ans=False, ansatz="walk", scale=1.0):
        self.m = WalkQNN(fm_steps, ans_steps, complex_fm, complex_ans,
                         ansatz, scale)
        self.g = self.m.g
        self.n = self.g.n_qubits
        self.n_weights = self.m.n_weights
        self.signs = self.g.signs

    def encode(self, X):
        out = np.empty((len(X), self.g.n_pad, self.g.d), dtype=complex)
        for i, x in enumerate(X):
            psi = self.g._psi0
            out[i] = self.g.evolve(psi, self.m._fm_stacks(x))
        return out

    def outputs(self, enc, w):
        if self.m.ansatz == "walk":
            psi = enc
            for C in self.m._ans_stacks(w):
                psi = np.einsum("vij,bvj->bvi", C, psi)
                flat = psi.reshape(len(enc), -1)
                nxt = np.empty_like(flat)
                nxt[:, self.g.perm] = flat
                psi = nxt.reshape(len(enc), self.g.n_pad, self.g.d)
            return (np.abs(psi.reshape(len(enc), -1)) ** 2) @ self.signs
        s = enc.reshape(len(enc), -1).copy()
        n = self.n
        for q in range(n):
            s = b_apply_1q(s, _ry(w[q]), q, n)
        for q in range(n - 1):
            s = b_apply_cnot(s, q, q + 1, n)
        for q in range(n):
            s = b_apply_1q(s, _rx(w[n + q]), q, n)
        return (np.abs(s) ** 2) @ self.signs


# ----------------------------------------------------------------- train --


def train_once(runner, enc_tr, ytr, enc_te, yte, seed, maxiter=MAXITER):
    rng = np.random.default_rng(seed)
    w0 = rng.uniform(0, 2 * np.pi, runner.n_weights)

    hist = []

    def loss(w):
        f = runner.outputs(enc_tr, w)
        val = float(np.mean((f - ytr) ** 2))
        hist.append(val)
        return val

    res = minimize(loss, w0, method="COBYLA", options={"maxiter": maxiter})
    w = res.x
    acc_tr = float(np.mean(np.sign(runner.outputs(enc_tr, w)) == ytr))
    acc_te = float(np.mean(np.sign(runner.outputs(enc_te, w)) == yte))
    return acc_tr, acc_te, float(res.fun), hist


def evaluate(name, runner, Xtr, ytr, Xte, yte, n_seeds=N_SEEDS):
    t0 = time.time()
    enc_tr, enc_te = runner.encode(Xtr), runner.encode(Xte)
    tr, te, losses, hists = [], [], [], []
    for s in range(n_seeds):
        a_tr, a_te, lo, h = train_once(runner, enc_tr, ytr, enc_te, yte, 1000 + s)
        tr.append(a_tr); te.append(a_te); losses.append(lo); hists.append(h)
    return dict(
        name=name, qubits=runner.n, weights=runner.n_weights,
        train_acc_mean=float(np.mean(tr)), train_acc_std=float(np.std(tr)),
        test_acc_mean=float(np.mean(te)), test_acc_std=float(np.std(te)),
        test_acc_best=float(np.max(te)), test_acc_worst=float(np.min(te)),
        final_loss_mean=float(np.mean(losses)),
        seconds=round(time.time() - t0, 1),
        histories=hists,
    )


# ------------------------------------------------------------------ main --


def build_models():
    hp = horizontal_pairs()
    poor = [(0, 1), (1, 2), (2, 3)]
    return [
        # the course's own two models -- the second is its headline result and
        # the reference point this pipeline is validated against
        ("course ZFeatureMap + poor ansatz", GateRunner(poor)),
        ("course ZFeatureMap + full ansatz", GateRunner(hp)),
        # walk feature map, walk ansatz: real vs complex coins
        ("walk FM (real) + walk ansatz (real)",
         WalkRunner(2, 2, False, False, "walk")),
        ("walk FM (cplx) + walk ansatz (real)",
         WalkRunner(2, 2, True, False, "walk")),
        ("walk FM (real) + walk ansatz (cplx)",
         WalkRunner(2, 2, False, True, "walk")),
        ("walk FM (cplx) + walk ansatz (cplx)",
         WalkRunner(2, 2, True, True, "walk")),
        # Equal-parameter controls.  A complex coin carries two numbers per
        # vertex where a real one carries one, so any complex "win" is a
        # parameter-count win until a real model with the same budget is shown
        # to fall short.  Four real steps = 32 weights = a two-step complex
        # ansatz, with the feature map held fixed on each side.
        ("walk FM (real) + walk ansatz (real, 4 steps)",
         WalkRunner(2, 4, False, False, "walk")),
        ("walk FM (cplx) + walk ansatz (real, 4 steps)",
         WalkRunner(2, 4, True, False, "walk")),
        # walk feature map with the course's style of ansatz, to separate the
        # feature map's contribution from the ansatz's
        ("walk FM (real) + rotation ansatz",
         WalkRunner(2, 2, False, False, "gate")),
        ("walk FM (cplx) + rotation ansatz",
         WalkRunner(2, 2, True, False, "gate")),
    ]


def verify_batched():
    """The batched path must equal the per-sample forward it replaces."""
    Xtr, ytr, _Xte, _yte = load()
    rng = np.random.default_rng(3)
    worst = 0.0
    for name, runner in build_models():
        if isinstance(runner, GateRunner):
            ref = GateQNN(runner.pairs)
        else:
            ref = runner.m
        w = rng.uniform(0, 2 * np.pi, runner.n_weights)
        enc = runner.encode(Xtr[:6])
        batched = runner.outputs(enc, w)
        single = np.array([ref.forward(x, w) for x in Xtr[:6]])
        worst = max(worst, float(np.abs(batched - single).max()))
        print(f"  {'OK ' if np.allclose(batched, single, atol=1e-11) else 'BAD'} "
              f"{name:<46} max |delta| = {np.abs(batched - single).max():.2e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--maxiter", type=int, default=MAXITER)
    args = ap.parse_args()

    if args.verify:
        print("Verifying the batched path against the per-sample forward:")
        w = verify_batched()
        print(f"\nworst deviation over all models: {w:.2e}")
        raise SystemExit(0 if w < 1e-11 else 1)

    MAXITER = args.maxiter
    Xtr, ytr, Xte, yte = load()
    print(f"IBM line-detection task: {len(ytr)} train, {len(yte)} test, "
          f"{args.seeds} seeds, COBYLA maxiter {args.maxiter}\n")

    # How hard is this task actually?  Without this, a quantum model scoring
    # 98% sounds impressive when a linear classifier on the raw pixels may
    # already be at ceiling, in which case the comparison is about nothing.
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.dummy import DummyClassifier

    print("classical reference points on the same split:")
    for cname, clf in [
        ("majority class", DummyClassifier(strategy="most_frequent")),
        ("logistic regression", LogisticRegression(max_iter=5000)),
        ("SVM, linear kernel", SVC(kernel="linear")),
        ("SVM, RBF kernel", SVC(kernel="rbf")),
    ]:
        clf.fit(Xtr, ytr)
        print(f"  {cname:<22} test {clf.score(Xte, yte) * 100:5.1f}   "
              f"train {clf.score(Xtr, ytr) * 100:5.1f}")
    print()

    rows = []
    for name, runner in build_models():
        r = evaluate(name, runner, Xtr, ytr, Xte, yte, args.seeds)
        rows.append(r)
        print(f"{r['name']:<46} q={r['qubits']:<2} p={r['weights']:<3} "
              f"test {r['test_acc_mean'] * 100:5.1f} +/- {r['test_acc_std'] * 100:4.1f} "
              f"(worst {r['test_acc_worst'] * 100:5.1f}, best {r['test_acc_best'] * 100:5.1f})  "
              f"train {r['train_acc_mean'] * 100:5.1f}  [{r['seconds']}s]")

    with open("qnn_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote qnn_results.json")
