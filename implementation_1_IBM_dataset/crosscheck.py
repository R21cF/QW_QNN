"""The numpy cores must agree with Qiskit, or the speed they buy is worthless."""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import z_feature_map as qk_zfm, UnitaryGate
from qiskit.quantum_info import Statevector, SparsePauliOp

from ibm_dataset import load, horizontal_pairs, N_PIXELS
from qnn_core import (
    GateQNN, WalkQNN, GridWalk, z_feature_map, ibm_ansatz, parity_signs,
    _ry, _rx, apply_1q, apply_cnot,
)

RNG = np.random.default_rng(5)
results = []


def check(name, cond, extra=""):
    results.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))


Xtr, ytr, Xte, yte = load()
x = Xtr[0]

# ------------------------------------------------------------------ 1 -----
print("\n[1] ZFeatureMap encoding matches Qiskit")

for reps in (1, 2):
    qc = qk_zfm(feature_dimension=N_PIXELS, reps=reps)
    bound = qc.assign_parameters(x)
    qk = np.asarray(Statevector(bound).data)
    mine = z_feature_map(x, N_PIXELS, reps=reps)
    err = np.abs(qk - mine).max()
    check(f"z_feature_map reps={reps}", err < 1e-12, f"max |delta| = {err:.2e}")

# ------------------------------------------------------------------ 2 -----
print("\n[2] full gate model matches a Qiskit circuit")

pairs = horizontal_pairs()
w = RNG.uniform(0, 2 * np.pi, 2 * N_PIXELS)

qc = QuantumCircuit(N_PIXELS)
qc.compose(qk_zfm(feature_dimension=N_PIXELS, reps=2).assign_parameters(x),
           inplace=True)
for q in range(N_PIXELS):
    qc.ry(w[q], q)
for a, b in pairs:
    qc.cx(a, b)
for q in range(N_PIXELS):
    qc.rx(w[N_PIXELS + q], q)

qk_state = np.asarray(Statevector(qc).data)
mine_state = ibm_ansatz(z_feature_map(x, N_PIXELS, 2), w, N_PIXELS, pairs)
err = np.abs(qk_state - mine_state).max()
check("gate model statevector", err < 1e-12, f"max |delta| = {err:.2e}")

obs = SparsePauliOp("Z" * N_PIXELS)
qk_exp = float(np.real(Statevector(qc).expectation_value(obs)))
mine_exp = GateQNN(pairs).forward(x, w)
check("gate model <Z...Z>", abs(qk_exp - mine_exp) < 1e-12,
      f"qiskit {qk_exp:+.10f}  numpy {mine_exp:+.10f}")

# ------------------------------------------------------------------ 3 -----
print("\n[3] walk operators are unitary and norm-preserving")

g = GridWalk()
print(f"        grid graph: {g.n_nodes} nodes, degrees {list(g.degrees)}, "
      f"coin dim {g.d}, {g.n_qubits} qubits ({g.n_pos_q}+{g.n_coin_q}), dim {g.dim}")

C = g.coin_stack(RNG.uniform(0, np.pi, g.n_nodes),
                 RNG.uniform(0, np.pi, g.n_nodes))
worst = max(np.abs(C[v].conj().T @ C[v] - np.eye(g.d)).max()
            for v in range(g.n_nodes))
check("every per-vertex coin is unitary", worst < 1e-12, f"max |delta| = {worst:.2e}")
check("shift is a permutation", sorted(g.perm.tolist()) == list(range(g.dim)))

psi = g._psi0
check("initial state normalised", abs(1 - np.linalg.norm(psi)) < 1e-12)
check("initial state puts no amplitude on padding directions",
      all(abs(psi[v, d]) < 1e-15
          for v in range(g.n_nodes) for d in range(int(g.degrees[v]), g.d)))

for _t in range(6):
    psi = g.step(psi, C)
check("norm conserved over 6 steps", abs(1 - np.linalg.norm(psi)) < 1e-12,
      f"|1 - norm| = {abs(1 - np.linalg.norm(psi)):.2e}")

# ------------------------------------------------------------------ 4 -----
print("\n[4] walk model matches an equivalent Qiskit circuit")


def walk_circuit(model, x, w):
    """Same computation, built as a Qiskit circuit of dense step unitaries."""
    gg = model.g
    n = gg.n_qubits
    qc = QuantumCircuit(n)
    qc.initialize(gg._psi0.reshape(-1), list(qc.qubits))
    stacks = list(model._fm_stacks(x)) + (
        list(model._ans_stacks(w)) if model.ansatz == "walk" else [])
    perm_mat = np.zeros((gg.dim, gg.dim))
    perm_mat[gg.perm, np.arange(gg.dim)] = 1.0
    for k, C in enumerate(stacks):
        big = np.zeros((gg.dim, gg.dim), dtype=complex)
        for v in range(gg.n_pad):
            big[v * gg.d:(v + 1) * gg.d, v * gg.d:(v + 1) * gg.d] = C[v]
        qc.append(UnitaryGate(perm_mat @ big, label=f"W{k}"), list(qc.qubits))
    return qc


for cfm, cans in [(False, False), (True, True)]:
    m = WalkQNN(fm_steps=2, ans_steps=2, complex_fm=cfm, complex_ans=cans)
    wv = RNG.uniform(0, 2 * np.pi, m.n_weights)
    qc = walk_circuit(m, x, wv)
    sv = Statevector(qc)
    qk_exp = float(np.real(sv.expectation_value(SparsePauliOp("Z" * m.n))))
    mine_exp = m.forward(x, wv)
    check(f"walk model <Z...Z> (complex={cfm})", abs(qk_exp - mine_exp) < 1e-10,
          f"qiskit {qk_exp:+.10f}  numpy {mine_exp:+.10f}")

    tq = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3,
                   seed_transpiler=7)
    print(f"        complex={cfm}: {m.n} qubits, transpiled depth {tq.depth()}, "
          f"{tq.count_ops().get('cx', 0)} CX")

# baseline depth for comparison
qc_base = QuantumCircuit(N_PIXELS)
qc_base.compose(qk_zfm(feature_dimension=N_PIXELS, reps=2).assign_parameters(x),
                inplace=True)
for q in range(N_PIXELS):
    qc_base.ry(w[q], q)
for a, b in pairs:
    qc_base.cx(a, b)
for q in range(N_PIXELS):
    qc_base.rx(w[N_PIXELS + q], q)
tb = transpile(qc_base, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3,
               seed_transpiler=7)
print(f"        course baseline: {N_PIXELS} qubits, transpiled depth {tb.depth()}, "
      f"{tb.count_ops().get('cx', 0)} CX")

# ------------------------------------------------------------------------
n_pass = sum(1 for _n, c in results if c)
print(f"\n{'=' * 60}\n{n_pass}/{len(results)} checks passed\n{'=' * 60}")
if n_pass != len(results):
    raise SystemExit(1)
