"""Standard quantum-kernel baselines: the Z and ZZ feature maps."""
from __future__ import annotations
import numpy as np

__all__ = ["z_states", "zz_states", "z_kernel", "zz_kernel", "check"]

_CACHE = {}


def _basis(d):
    if d not in _CACHE:
        idx = np.arange(1 << d)
        bits = ((idx[:, None] >> np.arange(d)[None, :]) & 1)      # little-endian
        s = 1.0 - 2.0 * bits                                      # Z eigenvalues, +/-1
        a = idx[:, None] & idx[None, :]
        par = np.zeros_like(a)
        t = a.copy()
        while t.any():
            par ^= (t & 1); t >>= 1
        H = ((-1.0) ** par) / np.sqrt(1 << d)
        _CACHE[d] = (s, H)
    return _CACHE[d]


def _states(X, lam, reps, pairs):
    X = np.atleast_2d(np.asarray(X, float))
    m, d = X.shape
    s, H = _basis(d)                                   # s: (2^d, d)
    lin = lam * (X @ s.T)                              # (m, 2^d)
    phi = lin
    if pairs:
        quad = np.zeros_like(lin)
        for j in range(d):
            for k in range(j + 1, d):
                c = lam * (np.pi - X[:, j]) * (np.pi - X[:, k])       # (m,)
                quad += c[:, None] * (s[:, j] * s[:, k])[None, :]
        phi = lin + quad
    ph = np.exp(-1j * phi)   # P(2phi) = exp(-i phi z) up to a global phase
    psi = np.zeros((m, 1 << d), dtype=np.complex128)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = psi @ H.T
        psi = psi * ph
    return psi


def z_states(X, lam=1.0, reps=2):   return _states(X, lam, reps, False)
def zz_states(X, lam=1.0, reps=2):  return _states(X, lam, reps, True)


def _fid(A, B):
    return np.abs(A.conj() @ B.T) ** 2


def z_kernel(XA, XB=None, lam=1.0, reps=2):
    A = z_states(XA, lam, reps); B = A if XB is None else z_states(XB, lam, reps)
    return _fid(A, B)


def zz_kernel(XA, XB=None, lam=1.0, reps=2):
    A = zz_states(XA, lam, reps); B = A if XB is None else zz_states(XB, lam, reps)
    return _fid(A, B)


def check(seed=0):
    """Agreement with Qiskit's own ZFeatureMap / ZZFeatureMap at lam = 1."""
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
    from qiskit.quantum_info import Statevector
    rng = np.random.RandomState(seed)
    worst = {"z": 0.0, "zz": 0.0}
    for d in (2, 3, 4):
        for reps in (1, 2, 3):
            x = rng.uniform(-2, 2, d)
            for tag, FM, mine in (("z", ZFeatureMap, z_states), ("zz", ZZFeatureMap, zz_states)):
                qc = FM(feature_dimension=d, reps=reps).assign_parameters(x)
                a = np.asarray(Statevector(qc))
                b = mine(x[None, :], 1.0, reps)[0]
                # global phase is unobservable in a fidelity kernel
                ov = abs(np.vdot(a, b))
                worst[tag] = max(worst[tag], abs(1.0 - ov))
    return worst


if __name__ == "__main__":
    w = check()
    print(f"max |1 - |<qiskit|mine>|| :  Z {w['z']:.3e}   ZZ {w['zz']:.3e}")
    print("PASS" if max(w.values()) < 1e-10 else "FAIL")
