"""The same feature map as a Qiskit circuit.

walkmap.py is a fast statevector path used for the hyperparameter search; this is
the circuit that actually defines the map. `check()` asserts the two agree to
machine precision, so any number produced by the fast path is a number this
circuit would produce.

Qubit order: position qubits are q0..q_{n-1} (little-endian, q0 the LSB of the
site index) and the coin is q_n, so Statevector index = p + 2^n * c, matching
walkmap's psi[c, p].reshape(-1).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def _c_increment(qc, ctrl, pos):
    """Controlled +1 mod 2^n on a little-endian register."""
    n = len(pos)
    for k in range(n - 1, 0, -1):
        qc.mcx([ctrl] + list(pos[:k]), pos[k])
    qc.cx(ctrl, pos[0])


def _c_decrement(qc, ctrl, pos):
    """Controlled -1 mod 2^n: the increment's gates in reverse (each is self-inverse)."""
    n = len(pos)
    qc.cx(ctrl, pos[0])
    for k in range(1, n):
        qc.mcx([ctrl] + list(pos[:k]), pos[k])


def walk_circuit(x, lam, T, n, theta0=np.pi / 4, barriers=False):
    pos = QuantumRegister(n, "p")
    coin = QuantumRegister(1, "c")
    qc = QuantumCircuit(pos, coin)
    d = len(x)
    for t in range(T):
        th = theta0 + lam * x[t % d]
        qc.u(2 * th, 0.0, np.pi, coin[0])          # [[cos th, sin th], [sin th, -cos th]]
        _c_increment(qc, coin[0], pos)             # coin = 1  ->  p + 1
        qc.x(coin[0])
        _c_decrement(qc, coin[0], pos)             # coin = 0  ->  p - 1
        qc.x(coin[0])
        if barriers:
            qc.barrier()
    return qc


def circuit_state(x, lam, T, n, theta0=np.pi / 4):
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector(walk_circuit(x, lam, T, n, theta0)))


def fidelity_circuit(x, y, lam, T, n, theta0=np.pi / 4):
    """Compute-uncompute circuit: P(all zeros) = |<psi(x)|psi(y)>|^2."""
    qc = walk_circuit(y, lam, T, n, theta0)
    qc = qc.compose(walk_circuit(x, lam, T, n, theta0).inverse())
    qc.measure_all()
    return qc


def check(seed=0, tol=1e-10):
    import walkmap as W
    rng = np.random.RandomState(seed)
    worst = 0.0
    for n in (3, 4, 5):
        for T in (4, 7, 12):
            for d in (2, 3, 5):
                x = rng.randn(d)
                lam = rng.uniform(0.2, 1.5)
                a = circuit_state(x, lam, T, n)
                b = W.walk_state(x, lam, T, n)
                worst = max(worst, np.abs(a - b).max())
    return worst


if __name__ == "__main__":
    w = check()
    print(f"max |qiskit - numpy| over 27 configurations: {w:.3e}")
    print("PASS" if w < 1e-10 else "FAIL")
    qc = walk_circuit(np.array([0.3, -0.7]), 0.9, 4, 4)
    print(f"\ncircuit: {qc.num_qubits} qubits, depth {qc.depth()}, ops {dict(qc.count_ops())}")
    from qiskit import transpile
    t = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3)
    print(f"transpiled to CX basis: depth {t.depth()}, CX {t.count_ops().get('cx', 0)}")
