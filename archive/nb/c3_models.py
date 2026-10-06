# Exported from qwqnn_lat_colab.ipynb by nb/export_cells.py; do not edit here.
# ---- models ---------------------------------------------------------------------------
# Hadamard-coin QW-QNN. Input: the exponent a = log_g x (from the discrete-log step).
#   position register: h qubits holding the top h bits of a  ->  vertex v = floor(a 2^h/(p-1)) of the cycle C_{2^h}
#   coin: 1 qubit, initialised (|0> + i|1>)/sqrt2
#   t steps of  W = S (H_coin (x) I),  S = |0><0| (x) INC + |1><1| (x) DEC   (Hadamard walk on C_{2^h})
#   readout: measure position, f(a) = sum_u w_u P_t(u | v) + b   (trainable observable O_w = sum_u w_u |u><u|)
#   training: the IBM lesson's MSE loss on y in {-1,+1}; f is linear in (w, b), so the minimiser is exact (ridge)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC as _SVC

H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def walk_distribution(h, t):
    """P_t(u | start 0) for the Hadamard walk on C_{2^h}; translation invariance gives P_t(u|v) = P_t(u-v|0)."""
    N = 2 ** h
    psi = np.zeros((2, N), complex)
    psi[:, 0] = np.array([1, 1j]) / np.sqrt(2)
    for _ in range(t):
        psi = H2 @ psi
        psi = np.stack([np.roll(psi[0], 1), np.roll(psi[1], -1)])   # coin 0 -> v+1, coin 1 -> v-1
    return (np.abs(psi) ** 2).sum(0)


class HadamardWalkQNN(BaseEstimator, ClassifierMixin):
    """X: column vector of exponents a (ints). t = 0 is the no-walk control (readout of the start vertex only)."""

    def __init__(self, p=251, h=4, t=4, ridge=1e-2, shots=None, random_state=0):
        self.p, self.h, self.t, self.ridge, self.shots, self.random_state = p, h, t, ridge, shots, random_state

    def _probs(self, X):
        a = np.asarray(X).reshape(-1).astype(np.int64)
        N = 2 ** self.h
        v = (a * N) // (self.p - 1)
        P0 = walk_distribution(self.h, self.t)
        idx = (np.arange(N)[None, :] - v[:, None]) % N
        return P0[idx]                                           # (B, N): P_t(u | v)

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.classes_ = np.array([-1, 1])
        Phi = np.c_[self._probs(X), np.ones(len(y))]
        R = self.ridge * np.eye(Phi.shape[1]); R[-1, -1] = 0
        self.coef_ = np.linalg.solve(Phi.T @ Phi + R, Phi.T @ y)
        return self

    def decision_function(self, X):
        P = self._probs(X)
        if self.shots:                                           # finite-shot estimate of <O_w>
            rng = np.random.default_rng(self.random_state)
            P = np.array([rng.multinomial(self.shots, q / q.sum()) for q in P]) / self.shots
        return P @ self.coef_[:-1] + self.coef_[-1]

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class LATKernelSVM(BaseEstimator, ClassifierMixin):
    """Liu-Arunachalam-Temme quantum kernel: |phi(x)> = 2^{-k/2} sum_{i<2^k} |x g^i>,
    K = |<phi(x)|phi(x')>|^2 = (max(0, 2^k - d(a,a')) / 2^k)^2,  d = forward cyclic distance in exponents
    (the overlap of two length-2^k intervals depends on the circular distance)."""

    def __init__(self, p=251, k=6, C=1.0):
        self.p, self.k, self.C = p, k, C

    def _K(self, A, B):
        A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(1, -1)
        d = np.abs(A - B) % (self.p - 1)
        d = np.minimum(d, self.p - 1 - d)
        W = 2 ** self.k
        return (np.maximum(0, W - d) / W) ** 2

    def fit(self, X, y):
        self.X_ = np.asarray(X).reshape(-1)
        self.classes_ = np.array([-1, 1])
        self.svc_ = _SVC(kernel="precomputed", C=self.C).fit(self._K(self.X_, self.X_), y)
        return self

    def predict(self, X):
        return self.svc_.predict(self._K(X, self.X_))


def build_walk_circuit(h, t, v):
    """The walk as a Qiskit circuit (qubit 0 = coin, qubits 1..h = position, little-endian), for checking."""
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


# ---- continuous-time variant: CTQW on the same cycle, same read-out -------------------------------------
# |psi_t> = exp(-i t A) |v>,  A = adjacency of C_{2^h};  f(a) = sum_u w_u |<u|exp(-i t A)|v>|^2 + b
# exp(-itA) = F diag(exp(-2it cos(2 pi k/N))) F^dagger  (F = QFT): a QFT, a diagonal phase, an inverse QFT.
def ctqw_distribution(h, t):
    N = 2 ** h
    k = np.arange(N)
    lam = 2 * np.cos(2 * np.pi * k / N)
    amp = np.fft.ifft(np.exp(-1j * t * lam))          # <u|e^{-itA}|0> = (1/N) sum_k e^{2 pi i u k/N} e^{-it lam_k}
    return np.abs(amp) ** 2


class CTWalkQNN(HadamardWalkQNN):
    """Same model as HadamardWalkQNN with the coined walk replaced by exp(-i t A_{C_{2^h}}); t is continuous."""

    def _probs(self, X):
        a = np.asarray(X).reshape(-1).astype(np.int64)
        N = 2 ** self.h
        v = (a * N) // (self.p - 1)
        P0 = ctqw_distribution(self.h, self.t)
        return P0[(np.arange(N)[None, :] - v[:, None]) % N]

    def fit(self, X, y):
        # same least-squares problem (ridge on w, unpenalised bias); sklearn switches to the dual form when 2^h > samples
        from sklearn.linear_model import Ridge
        self.classes_ = np.array([-1, 1])
        r = Ridge(alpha=self.ridge).fit(self._probs(X), np.asarray(y, float))
        self.coef_ = np.r_[r.coef_, r.intercept_]
        return self


def build_ctqw_circuit(h, t, v):
    """exp(-itA) on the cycle as QFT^dagger . diag . QFT on h qubits (little-endian), for checking."""
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
