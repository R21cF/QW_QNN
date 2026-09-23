"""
Gradient variance of the walk ansatz against register width (graph size).

A barren plateau is exponential decay of Var[dC/dw] in the number of qubits.
For a walk on N vertices of degree <= d the register has log2 N + log2 d
qubits, so graph size is the width variable. Grids R x C with N = R*C a power
of two give registers of 5 to 12 qubits.

Walk ansatz (axis 1, per-vertex coin): the uniform arc start, then T = R + C steps (enough for amplitude
to cross the grid), each step a per-vertex real rotation coin
(givens_chain, independent random angle at every vertex and step) followed by
the flip-flop shift. No data: this isolates the ansatz, as in McClean et al.

Walk ansatz (axis 2, per-step coin): the same walk with one angle per step
shared by every vertex -- T parameters in all.

Baseline: a hardware-efficient ansatz (RY, CNOT ladder, RX) on the same number
of qubits and the same number of layers, from |0...0>.

Costs: global C = <Z on every qubit> (the classifier's observable) and local
C = <Z on the most significant qubit>. Global costs are known to flatten even
shallow circuits (Cerezo et al. 2021), so both are reported.

Gradients are central finite differences; for the walk only the one coin block
that a parameter controls is recomputed.
"""

from __future__ import annotations

import json
import time

import numpy as np

from qnn_core import GridWalk, givens_chain, parity_signs, _ry, _rx
from run_qnn import b_apply_1q, b_apply_cnot

GRIDS = [(2, 4), (4, 4), (4, 8), (8, 8), (8, 16), (16, 16), (16, 32), (32, 32)]
N_DRAWS = 200
PARAMS_PER_DRAW = 8
H = 1e-4
SEED = 7


def grid_edges(R, C):
    e = []
    for r in range(R):
        for c in range(C):
            v = r * C + c
            if c + 1 < C:
                e.append((v, v + 1))
            if r + 1 < R:
                e.append((v, v + C))
    return e


def observables(n):
    idx = np.arange(1 << n)
    glob = parity_signs(n)
    loc = 1.0 - 2.0 * ((idx >> (n - 1)) & 1)      # Z on the most significant qubit
    return glob, loc


class WalkAnsatz:
    def __init__(self, R, C):
        self.g = GridWalk(grid_edges(R, C), R * C)
        self.T = R + C
        self.N = R * C
        self.n = self.g.n_qubits
        self.n_weights = self.T * self.N
        self.glob, self.loc = observables(self.n)

    def stacks(self, w):
        return [self.g.coin_stack(w[t * self.N:(t + 1) * self.N]) for t in range(self.T)]

    def probs(self, st):
        psi = self.g.evolve(self.g._psi0, st)
        return np.abs(psi.reshape(-1)) ** 2

    def grad(self, w, st, k, obs):
        t, v = divmod(k, self.N)
        deg = int(self.g.degrees[v])
        out = []
        for sgn in (+1, -1):
            blk = st[t].copy()
            blk[v, :deg, :deg] = givens_chain(deg, float(w[k] + sgn * H))
            st2 = list(st); st2[t] = blk
            out.append(self.probs(st2) @ obs)
        return (out[0] - out[1]) / (2 * H)


class WalkShared(WalkAnsatz):
    """Step-dependent coin (axis 2): one angle per step, shared by every vertex."""

    def __init__(self, R, C):
        super().__init__(R, C)
        self.n_weights = self.T

    def stacks(self, w):
        return [self.g.coin_stack(np.full(self.N, w[t])) for t in range(self.T)]

    def grad(self, w, st, k, obs):
        out = []
        for sgn in (+1, -1):
            st2 = list(st); st2[k] = self.g.coin_stack(np.full(self.N, w[k] + sgn * H))
            out.append(self.probs(st2) @ obs)
        return (out[0] - out[1]) / (2 * H)


class HEAWidth:
    def __init__(self, n, layers):
        self.n, self.layers = n, layers
        self.n_weights = 2 * n * layers
        self.glob, self.loc = observables(n)
        self.psi0 = np.zeros((1, 1 << n), dtype=complex); self.psi0[0, 0] = 1.0

    def probs(self, w):
        s, n, k = self.psi0, self.n, 0
        for _ in range(self.layers):
            for q in range(n):
                s = b_apply_1q(s, _ry(w[k]), q, n); k += 1
            for q in range(n - 1):
                s = b_apply_cnot(s, q, q + 1, n)
            for q in range(n):
                s = b_apply_1q(s, _rx(w[k]), q, n); k += 1
        return np.abs(s[0]) ** 2

    def grad(self, w, k, obs):
        e = np.zeros_like(w); e[k] = np.pi / 2           # exact: Pauli/2 generators
        return 0.5 * (self.probs(w + e) @ obs - self.probs(w - e) @ obs)


def summarise(g, rng):
    var = float(g.var(axis=0).mean())
    boot = [g[rng.integers(0, len(g), len(g))].var(axis=0).mean() for _ in range(500)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return var, float(lo), float(hi)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    # validation: walk finite difference vs brute-force full rebuild, 2x4 grid
    wa = WalkAnsatz(2, 4)
    w = rng.uniform(0, 2 * np.pi, wa.n_weights)
    st = wa.stacks(w)
    worst = 0.0
    for k in rng.choice(wa.n_weights, 8, replace=False):
        e = np.zeros_like(w); e[k] = H
        brute = (wa.probs(wa.stacks(w + e)) @ wa.glob - wa.probs(wa.stacks(w - e)) @ wa.glob) / (2 * H)
        worst = max(worst, abs(brute - wa.grad(w, st, k, wa.glob)))
    print(f"single-block update vs full rebuild: max |delta| = {worst:.1e}")
    assert worst < 1e-9

    import os
    done = set()
    if os.path.exists("gradient_variance_width.rows.jsonl"):
        for line in open("gradient_variance_width.rows.jsonl"):
            r = json.loads(line); rows.append(r); done.add((r["ansatz"], r["grid"]))
    for R, C in GRIDS:
        wa = WalkAnsatz(R, C)
        hea = HEAWidth(wa.n, wa.T)
        for name, model in (("walk, per-vertex coin", wa),
                            ("walk, per-step coin", WalkShared(R, C)),
                            ("hardware-efficient", hea)):
            if (name, f"{R}x{C}") in done:
                continue
            # per-row seed: every row is reproducible on its own, resumed or not
            rng = np.random.default_rng([SEED, GRIDS.index((R, C)),
                                         ["walk, per-vertex coin", "walk, per-step coin",
                                          "hardware-efficient"].index(name)])
            t0 = time.time()
            G, L = [], []
            for _ in range(N_DRAWS):
                w = rng.uniform(0, 2 * np.pi, model.n_weights)
                ks = rng.choice(model.n_weights, min(PARAMS_PER_DRAW, model.n_weights),
                                replace=False)
                if name.startswith("walk"):
                    st = model.stacks(w)
                    G.append([model.grad(w, st, k, model.glob) for k in ks])
                    L.append([model.grad(w, st, k, model.loc) for k in ks])
                else:
                    G.append([model.grad(w, k, model.glob) for k in ks])
                    L.append([model.grad(w, k, model.loc) for k in ks])
            vg, glo, ghi = summarise(np.array(G), rng)
            vl, llo, lhi = summarise(np.array(L), rng)
            row = dict(ansatz=name, grid=f"{R}x{C}", N=R * C, qubits=wa.n, depth=wa.T,
                       n_weights=model.n_weights,
                       var_global=vg, global_ci=[glo, ghi],
                       var_local=vl, local_ci=[llo, lhi],
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            with open("gradient_variance_width.rows.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{name:22s} {R:2d}x{C:<2d} q={wa.n:2d} T={wa.T:2d}  "
                  f"Var global {vg:.3e}  local {vl:.3e}  {row['seconds']}s", flush=True)
    # exponential fits in qubit count: Var ~ b^{-n}
    fits = {}
    for name in ("walk, per-vertex coin", "walk, per-step coin", "hardware-efficient"):
        for key in ("var_global", "var_local"):
            q = np.array([r["qubits"] for r in rows if r["ansatz"] == name])
            v = np.array([r[key] for r in rows if r["ansatz"] == name])
            slope = np.polyfit(q, np.log2(v), 1)[0]
            fits[f"{name} {key}"] = float(slope)
            print(f"fit {name:22s} {key:10s}: log2 Var slope per qubit = {slope:+.2f}")
    json.dump({"validation_max_abs_delta": worst, "n_draws": N_DRAWS,
               "params_per_draw": PARAMS_PER_DRAW, "rows": rows,
               "log2_slope_per_qubit": fits},
              open("gradient_variance_width.json", "w"), indent=1)


if __name__ == "__main__":
    main()
