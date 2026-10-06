"""Qiskit checks for the QAOA mixers of thesis Section 3.6 (described in Section 4.2.1).

1. The coined mixer on J(8,4): 70 vertices in a 7-qubit position register and the 16-dimensional coin in a 4-qubit
   coin register, coin in the low-order qubits so |v>|c> has index 16 v + c. One step U = S (I (x) e^{-i beta C_G}) is
   assembled as a matrix and appended as one dense UnitaryGate; the cost layer is a DiagonalGate on the position
   register. A depth-p circuit is compared with the NumPy simulation run_dtqw of qaoa_walk.py.
2. The transverse-field mixer as RX gates against run_x.
3. Eq. (xy-johnson): (1/2) sum_{i<j} (X_i X_j + Y_i Y_j) restricted to span F equals the Johnson adjacency A_J, and
   the continuous-time mixer built from it in Qiskit agrees with run_ctqw.
"""
import os, sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from scipy.linalg import expm
from qiskit.quantum_info import Statevector, SparsePauliOp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qaoa_walk as Q

rng = np.random.default_rng(0)
n, F, d = Q.n, Q.F, Q.d_J
POS, COIN = 7, 4
assert d == 2 ** COIN and F <= 2 ** POS
C_f = rng.integers(0, 20, F).astype(float)                       # an arbitrary cost on the feasible strings
p = 3
params = rng.uniform(0, np.pi, 2 * p)
gam, bet = params[:p], params[p:]

# ---- 1. coined mixer --------------------------------------------------------------------------------------
def step_matrix(beta):
    """Dense 2^11 x 2^11 matrix of one coined step, built column by column from dtqw_step; identity off span F."""
    D = 2 ** (POS + COIN)
    U = np.eye(D, dtype=complex)
    for v in range(F):
        for c in range(d):
            e = np.zeros((F, d), complex); e[v, c] = 1
            col = np.zeros(D, complex); col[: F * d] = Q.dtqw_step(e, beta).reshape(-1)   # index 16 v + c
            U[:, 16 * v + c] = col
    return U


qc = QuantumCircuit(POS + COIN)
psi0 = np.zeros(2 ** (POS + COIN), complex); psi0[: F * d] = 1 / np.sqrt(F * d)   # uniform over arcs
for g, b in zip(gam, bet):
    phases = np.ones(2 ** POS, complex); phases[:F] = np.exp(-1j * g * C_f)
    qc.append(DiagonalGate(list(phases)), range(COIN, COIN + POS))                 # position = high-order qubits
    U = step_matrix(b)
    assert np.abs(U.conj().T @ U - np.eye(len(U))).max() < 1e-12
    qc.append(UnitaryGate(U), range(POS + COIN))
pq = Statevector(psi0).evolve(qc).probabilities().reshape(2 ** POS, d).sum(1)[:F]
err1 = np.abs(pq - Q.run_dtqw(params, C_f, p)).max()
print(f"coined mixer on J(8,4), p = {p}: max |P_qiskit - P_numpy| = {err1:.1e}")

# ---- 2. transverse-field mixer ----------------------------------------------------------------------------
C_all = rng.integers(0, 20, 2 ** n).astype(float)
qc = QuantumCircuit(n); qc.h(range(n))
for g, b in zip(gam, bet):
    qc.append(DiagonalGate(list(np.exp(-1j * g * C_all))), range(n))
    qc.rx(2 * b, range(n))                                                          # e^{-i b X}
err2 = np.abs(Statevector(qc).probabilities() - Q.run_x(params, C_all, p)).max()
print(f"X mixer, p = {p}: max |P_qiskit - P_numpy| = {err2:.1e}")

# ---- 3. XY mixer = Johnson-graph walk ---------------------------------------------------------------------
terms = [("XX", [i, j], 0.5) for i in range(n) for j in range(i + 1, n)] + \
        [("YY", [i, j], 0.5) for i in range(n) for j in range(i + 1, n)]
HXY = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
M = HXY.to_matrix()[np.ix_(Q.feas_states, Q.feas_states)]
err3 = np.abs(M - Q.A_J).max()
print(f"(1/2) sum (XX + YY) restricted to span F vs A_J: max difference = {err3:.1e}")
qc = QuantumCircuit(n)
init = np.zeros(2 ** n, complex); init[Q.feas_states] = 1 / np.sqrt(F)
C_full = np.zeros(2 ** n); C_full[Q.feas_states] = C_f
for g, b in zip(gam, bet):
    qc.append(DiagonalGate(list(np.exp(-1j * g * C_full))), range(n))
    qc.append(UnitaryGate(expm(-1j * b * HXY.to_matrix())), range(n))                # exact e^{-i b H_XY}
pr = Statevector(init).evolve(qc).probabilities()
err4 = max(np.abs(pr[Q.feas_states] - Q.run_ctqw(params, C_f, p)).max(), 1 - pr[Q.feas_states].sum())
print(f"continuous-time (XY) mixer, p = {p}: max |P_qiskit - P_numpy| = {err4:.1e}")

assert max(err1, err2, err3, err4) < 1e-10
