"""Properties of the continuous-time walk used by the CTQW-QNN (numbers quoted in the thesis).
Single walker, H = A(C_N), N = 2^h, binary (log N-qubit) position encoding, start |v>, position measurement."""
import itertools
import numpy as np
from scipy.linalg import expm


def entropy(psi, h, qubits):
    """von Neumann entropy (bits) of the listed qubits; qubit q is bit q of the vertex index (little-endian)."""
    T = psi.reshape([2] * h)                         # axis 0 = most significant bit = qubit h-1
    ax = [h - 1 - q for q in qubits]
    M = np.transpose(T, ax + [a for a in range(h) if a not in ax]).reshape(2 ** len(ax), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log2(s)).sum())


rows = []
for h in [3, 4, 5, 6]:
    N = 2 ** h
    A = np.roll(np.eye(N), 1, 0) + np.roll(np.eye(N), -1, 0)
    L = 2 * np.eye(N) - A
    for t in [0.1, 0.25, 0.5, 1.5, 4.0, 8.0]:
        U = expm(-1j * t * A)
        P = np.abs(U[:, 0]) ** 2
        cuts = [c for r in range(1, h // 2 + 1) for c in itertools.combinations(range(h), r)]
        smax = max(entropy(U[:, v], h, list(c)) for v in range(N) for c in cuts)
        sym = np.abs(P - np.roll(P[::-1], 1)).max()                        # P(u) = P(-u)
        lap = np.abs(np.abs(expm(-1j * t * L)) ** 2 - np.abs(U) ** 2).max()  # adjacency vs Laplacian walk
        sd = np.sqrt((P * np.minimum(np.arange(N), N - np.arange(N)) ** 2).sum())
        rows.append((h, t, P[0], P[1], sd, smax, sym, lap))
        print(f"h={h} t={t:4}: P(stay)={P[0]:.4f} P(+1)={P[1]:.4f} spread sd={sd:.3f} | "
              f"max entanglement over all cuts and start vertices = {smax:.3f} bits (max possible {h // 2}) | "
              f"reflection asym {sym:.1e} | |P_L - P_A| {lap:.1e}")
