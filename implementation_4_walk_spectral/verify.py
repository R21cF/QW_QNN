"""Checks, before any benchmark number is trusted:
 1. the QFT diagonalises the cycle CTQW Hamiltonian;
 2. O_w commutes with the torus CTQW e^{-iHt} (f_w is a conserved quantity of the walk);
 3. the Qiskit circuit (StatePreparation -> QFT (x) QFT -> measure) reproduces
    the NumPy features exactly;
 4. noise-free bars/stripes are separated by the analytic observable
    w = +1 on (a=0, b!=0)... i.e. Theorem 2;
 5. transpiled two-qubit gate counts per stage.
"""
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, QFTGate
from qiskit.quantum_info import Statevector

from qwnn import cycle_adjacency, momentum_bin, walk_spectral_probs, bin_probs
from data import load

rng = np.random.default_rng(0)

# 1 ---------------------------------------------------------------------------
for L in [4, 8, 16, 32]:
    A = cycle_adjacency(L)
    F = np.fft.fft(np.eye(L), norm="ortho")
    D = F @ A @ F.conj().T
    off = np.abs(D - np.diag(np.diag(D))).max()
    ev = np.sort(np.diag(D).real)
    ok = np.allclose(ev, np.sort(2 * np.cos(2 * np.pi * np.arange(L) / L)))
    print(f"[1] L={L}: max off-diagonal of F A F^dag = {off:.1e}; eigenvalues 2cos(2pi k/L): {ok}")

# 2 ---------------------------------------------------------------------------
L = 8
A = cycle_adjacency(L); I = np.eye(L)
H = np.kron(A, I) + np.kron(I, A)
F = np.fft.fft(np.eye(L), norm="ortho")
FF = np.kron(F, F)
a = momentum_bin(L)
w = rng.normal(size=(L // 2 + 1, L // 2 + 1))
O = FF.conj().T @ np.diag([w[a[i], a[j]] for i in range(L) for j in range(L)]) @ FF
U = expm(-1j * 0.73 * H)
print(f"[2] ||[O_w, e^(-iHt)]|| = {np.abs(O @ U - U @ O).max():.1e};  O_w Hermitian: {np.allclose(O, O.conj().T)}")

# 3 + 5 -----------------------------------------------------------------------
for L in [4, 8, 16, 32]:
    m = int(np.log2(L))
    Xtr, ytr, Xte, yte = load(L)
    x = Xte[0]
    qc = QuantumCircuit(2 * m)
    qc.append(StatePreparation(x / np.linalg.norm(x)), range(2 * m))
    qc.append(QFTGate(m), range(0, m))          # column register (little-endian LSBs)
    qc.append(QFTGate(m), range(m, 2 * m))      # row register
    if L <= 16:
        p = Statevector(qc).probabilities()      # index = r*L + c
        Pq = p.reshape(L, L)
        Pn = walk_spectral_probs(x[None], L)[0]
        err = np.abs(bin_probs(Pq[None], L) - bin_probs(Pn[None], L)).max()
        print(f"[3] L={L}: max |binned circuit probs - NumPy features| = {err:.1e}")
    basis = ["cx", "rz", "sx", "x"]
    prep = QuantumCircuit(2 * m); prep.append(StatePreparation(x / np.linalg.norm(x)), range(2 * m))
    walk = QuantumCircuit(2 * m); walk.append(QFTGate(m), range(0, m)); walk.append(QFTGate(m), range(m, 2 * m))
    tp = transpile(prep, basis_gates=basis, optimization_level=3)
    tw = transpile(walk, basis_gates=basis, optimization_level=3)
    print(f"[5] L={L}: qubits={2*m}; state prep CX={tp.count_ops().get('cx',0)} depth={tp.depth()}; "
          f"walk-eigenbasis layer CX={tw.count_ops().get('cx',0)} depth={tw.depth()}")

# 4 ---------------------------------------------------------------------------
for L in [4, 8, 16, 32]:
    nb = L // 2 + 1
    w = np.zeros((nb, nb)); w[0, 1:] = -1.0; w[1:, 0] = +1.0   # Pi_{row k=0} - Pi_{col k=0}, DC removed
    fails = 0; tested = 0
    for _ in range(2000):
        if rng.random() > 0.5:   # stripe: rows constant
            s = np.where(rng.random(L) > 0.5, 1.0, -1.0); img = np.repeat(s[:, None], L, 1); y = 1
        else:                    # bar: columns constant
            s = np.where(rng.random(L) > 0.5, 1.0, -1.0); img = np.repeat(s[None, :], L, 0); y = -1
        if np.all(s == s[0]):
            continue             # uniform image: identical for both labels
        tested += 1
        f = (bin_probs(walk_spectral_probs(img.reshape(1, -1), L), L)[0] * w.ravel()).sum()
        fails += np.sign(f) != y
    print(f"[4] L={L}: analytic observable on {tested} noise-free non-uniform images: {fails} errors")
