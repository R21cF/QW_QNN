"""Walks on the cycle C_N, N = 2^h, as NumPy computations and as Qiskit circuits (thesis Section 4.2.1).

Continuous-time walk:  e^{-itA} = F^dag diag(e^{-2it cos(2 pi k/N)}) F  (Lemma circulant), A the adjacency of C_N.
Coined Hadamard walk:  t steps of W = S (H (x) I), S = |0><0| (x) INC + |1><1| (x) DEC, coin (|0> + i|1>)/sqrt 2.
Both are translation invariant, so P_t(u | v) = P_t(u - v | 0) and one distribution per (h, t) serves every start.
"""
import numpy as np

H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def ctqw_distribution(h, t):
    """P_t(u | 0) = |<u| e^{-itA} |0>|^2 on C_{2^h}, by the fast Fourier transform."""
    N = 2 ** h
    k = np.arange(N)
    lam = 2 * np.cos(2 * np.pi * k / N)
    amp = np.fft.ifft(np.exp(-1j * t * lam))          # <u|e^{-itA}|0> = (1/N) sum_k e^{2 pi i u k/N} e^{-it lam_k}
    return np.abs(amp) ** 2


def hadamard_distribution(h, t):
    """P_t(u | 0) for t steps of the Hadamard walk on C_{2^h} (coin 0 -> v+1, coin 1 -> v-1)."""
    N = 2 ** h
    psi = np.zeros((2, N), complex)
    psi[:, 0] = np.array([1, 1j]) / np.sqrt(2)
    for _ in range(t):
        psi = H2 @ psi
        psi = np.stack([np.roll(psi[0], 1), np.roll(psi[1], -1)])
    return (np.abs(psi) ** 2).sum(0)


def cycle_adjacency(N):
    return np.roll(np.eye(N), 1, 0) + np.roll(np.eye(N), -1, 0)


def build_ctqw_circuit(h, t, v):
    """e^{-itA} on C_{2^h} started at |v>: QFT^dag, an explicit diagonal of the 2^h phases, QFT
    (little-endian). The explicit diagonal makes this a check circuit for small h only."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate, DiagonalGate
    N = 2 ** h
    qc = QuantumCircuit(h)
    for j in range(h):
        if (v >> j) & 1:
            qc.x(j)
    qc.append(QFTGate(h).inverse(), range(h))
    qc.append(DiagonalGate(list(np.exp(-1j * t * 2 * np.cos(2 * np.pi * np.arange(N) / N)))), range(h))
    qc.append(QFTGate(h), range(h))
    return qc


def build_hadamard_circuit(h, t, v):
    """The coined Hadamard walk: qubit 0 = coin, qubits 1..h = position (little-endian);
    increment and decrement circuits on the position register controlled on the coin."""
    from qiskit import QuantumCircuit
    inc = QuantumCircuit(h, name="INC")
    for j in reversed(range(1, h)):
        inc.mcx(list(range(j)), j)
    inc.x(0)
    INC0 = inc.to_gate().control(1, ctrl_state=0)
    DEC1 = inc.inverse().to_gate().control(1, ctrl_state=1)
    qc = QuantumCircuit(1 + h)
    for j in range(h):
        if (v >> j) & 1:
            qc.x(1 + j)
    qc.h(0); qc.s(0)                       # coin (|0> + i|1>)/sqrt2
    for _ in range(t):
        qc.h(0)
        qc.append(INC0, list(range(1 + h)))
        qc.append(DEC1, list(range(1 + h)))
    return qc


def check_ctqw(hs=(3, 4, 5), ts=(0.0, 0.5, 1.7, 4.0, 9.3)):
    """max |P| difference between the NumPy walk, SciPy's expm(-itA) and the Qiskit circuit."""
    from scipy.linalg import expm
    from qiskit.quantum_info import Statevector
    err = 0.0
    for h in hs:
        N = 2 ** h
        A = cycle_adjacency(N)
        for t in ts:
            for v in (0, 5, N - 1):
                pe = np.abs(expm(-1j * t * A)[:, v]) ** 2
                pn = ctqw_distribution(h, t)[(np.arange(N) - v) % N]
                pq = Statevector(build_ctqw_circuit(h, t, v)).probabilities()
                err = max(err, np.abs(pe - pn).max(), np.abs(pe - pq).max())
    return err


def check_hadamard(hs=(3, 4, 5), ts=(0, 1, 3, 7, 12)):
    """max |P| difference between the NumPy Hadamard walk and its Qiskit circuit."""
    from qiskit.quantum_info import Statevector
    err = 0.0
    for h in hs:
        N = 2 ** h
        for t in ts:
            for v in (0, 5, N - 1):
                pq = Statevector(build_hadamard_circuit(h, t, v)).probabilities().reshape(N, 2).sum(1)
                pn = hadamard_distribution(h, t)[(np.arange(N) - v) % N]
                err = max(err, np.abs(pq - pn).max())
    return err


def entanglement_entropy(psi, h, qubits):
    """von Neumann entropy (bits) of the listed position qubits; qubit q is bit q of the vertex index."""
    T = psi.reshape([2] * h)                         # axis 0 = most significant bit = qubit h-1
    ax = [h - 1 - q for q in qubits]
    M = np.transpose(T, ax + [a for a in range(h) if a not in ax]).reshape(2 ** len(ax), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log2(s)).sum())
