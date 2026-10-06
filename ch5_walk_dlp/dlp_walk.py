"""Discrete logarithm by phase estimation on walks over Z_p^* (Shor's DLP algorithm in walk form).

Walk operators on the work register |y>, y in Z_p^*:
  DT (coinless permutation walk, Prop. permwalk):  U_g |y> = |g y mod p>,  U_x |y> = |x y mod p>.
      On the orbit of |1> (all of Z_p^*, since g generates) both are cyclic shifts of the cycle C_r, r = p-1,
      diagonal in |psi_k> = r^{-1/2} sum_j w^{-jk} |g^j>, with U_g -> w^k and U_x = U_g^a -> w^{ka}, w = e^{2 pi i / r}.
  CT (continuous-time walk on the same Cayley graph):  H_g = U_g + U_g^dagger,  H_x = U_x + U_x^dagger,
      eigenvalues 2 cos(2 pi k / r) and 2 cos(2 pi k a / r) on the same |psi_k>.

Algorithm (both): work register |1> = r^{-1/2} sum_k |psi_k>; two m-qubit counting registers; controlled powers
  DT: c-U_g^{2^j}, c-U_x^{2^j}   (each one modular multiplication by g^{2^j} or x^{2^j}, precomputed classically)
  CT: c-exp(-i H_g tau 2^j), c-exp(-i H_x tau 2^j)   (tau = 7 pi/16 maps the spectrum [-2, 2] onto phases [-7/16, 7/16])
then inverse QFT on each counting register and measurement of (c1, c2).
(tau = 7 pi/16 < pi/2 keeps the CT phases -(tau/pi) cos inside (-1/2, 1/2); at tau = pi/2 the eigenvalues +2 and -2
map to phases +1/2 and -1/2, which alias, and x = -1 (a = r/2) could not be decoded.) The work states |psi_k> are orthogonal, so
  P(c1, c2) = (1/r) sum_k |F_M(phi1_k - c1/M)|^2 |F_M(phi2_k - c2/M)|^2,   F_M(d) = M^{-1} sum_{j<M} e^{2 pi i j d},  M = 2^m,
with phi1_k = k/r, phi2_k = ka/r (DT) or phi1_k = -(tau/pi) cos(2 pi k/r), phi2_k = -(tau/pi) cos(2 pi k a/r) (CT).
The distribution is checked against a Qiskit state-vector simulation of the circuits for small p (check_against_qiskit).

Decoding (r = p - 1 is known):
  DT: k~ = round(c1 r / M), l~ = round(c2 r / M); if gcd(k~, r) = 1, a~ = l~ k~^{-1} mod r.
  CT: the cosine is even in k, so c1 fixes k up to sign and c2 fixes ka up to sign: every k with
      cos(2 pi k/r) nearest the measured value, every l likewise, candidates a~ = l k^{-1} for gcd(k, r) = 1.
  Every candidate is verified classically with one modular exponentiation, g^{a~} = x (mod p); an attempt succeeds
  if a candidate verifies. With verification the procedure is repeated until success, so its output is exact.

Cost model, in oracle calls (one call = one controlled modular multiplication, or one unit of walk time):
  DT: 2m controlled multiplications per attempt (U^{2^j} is itself one multiplication).
  CT: exp(-i H tau 2^j) cannot be obtained by squaring; simulating H = U + U^dagger for time T needs Omega(T)
      calls to U (no-fast-forwarding), so an attempt costs about 2 tau (M - 1) calls.
"""
import numpy as np
from math import gcd

TAU = 7 * np.pi / 16


def fejer_offsets(frac, M, J):
    """P(c = round(M phi) + j) for j in [-J, J] given the fractional offset frac = M phi - round(M phi) (vector over samples)."""
    j = np.arange(-J, J + 1)
    d = (frac[:, None] - j[None, :]) / M                    # phi - c/M
    num = np.sin(np.pi * M * d) ** 2
    den = (M * np.sin(np.pi * d)) ** 2
    P = np.where(np.abs(den) < 1e-300, 1.0, num / np.where(den == 0, 1, den))
    P[np.abs(d) < 1e-15] = 1.0
    return j, P / P.sum(1, keepdims=True)


def sample_counts(phi, M, rng, J=4000):
    """Sample the measured integer c in [0, M) for each phase phi (array)."""
    base = np.round(phi * M)
    frac = phi * M - base
    j, P = fejer_offsets(frac, M, J)
    u = rng.random(len(phi))
    idx = (P.cumsum(1) < u[:, None]).sum(1)
    return ((base + j[np.minimum(idx, len(j) - 1)]) % M).astype(np.int64)


def _decode_dt(c1, c2, M, r):
    k = int(np.round(c1 * r / M)) % r
    l = int(np.round(c2 * r / M)) % r
    return [l * pow(k, -1, r) % r] if gcd(k, r) == 1 else []


def _decode_ct(c1, c2, M, r, cos_table):
    out = []
    for c, lst in ((c1, []), (c2, [])):
        ph = c / M if c < M // 2 else c / M - 1              # phase in [-1/2, 1/2)
        cest = -np.pi * ph / TAU                              # estimate of cos(2 pi k / r)
        dist = np.abs(cos_table - cest)
        best = dist.min()
        lst.extend(np.nonzero(dist <= best + 1e-12)[0].tolist())
        out.append(lst)
    ks, ls = out
    return [l * pow(k, -1, r) % r for k in ks if gcd(k, r) == 1 for l in ls]


def attempt(kind, a, p, g, m, rng, n_attempts):
    """n_attempts independent runs for one input with log a; returns boolean success array (after verification)."""
    r = p - 1; M = 2 ** m
    x = pow(g, int(a), p)
    k = rng.integers(0, r, n_attempts)                        # which eigenvector the work register collapses to
    if kind == "DT":
        phi1, phi2 = k / r, (k * a % r) / r
    else:
        phi1, phi2 = -(TAU / np.pi) * np.cos(2 * np.pi * k / r), -(TAU / np.pi) * np.cos(2 * np.pi * (k * a % r) / r)
    c1, c2 = sample_counts(phi1 % 1.0, M, rng), sample_counts(phi2 % 1.0, M, rng)
    cos_table = np.cos(2 * np.pi * np.arange(r) / r) if kind == "CT" else None
    ok = np.zeros(n_attempts, bool)
    for i in range(n_attempts):
        cands = _decode_dt(c1[i], c2[i], M, r) if kind == "DT" else _decode_ct(c1[i], c2[i], M, r, cos_table)
        ok[i] = any(pow(g, int(c), p) == x for c in cands)
    return ok


def calls_per_attempt(kind, m):
    return 2 * m if kind == "DT" else float(2 * TAU * (2 ** m - 1))


def solve_dlp(a, p, g, m, rng, max_attempts=200, verify=True):
    """DT walk DLP for one input. verify=True: repeat until a verified logarithm (exact output).
    verify=False: single attempt; returns the decoded candidate (or a uniformly random exponent if decoding fails)."""
    r = p - 1; M = 2 ** m; x = pow(g, int(a), p)
    for t in range(1, max_attempts + 1):
        k = int(rng.integers(0, r))
        c1 = int(sample_counts(np.array([k / r]), M, rng)[0]); c2 = int(sample_counts(np.array([(k * a % r) / r]), M, rng)[0])
        cands = _decode_dt(c1, c2, M, r)
        if not verify:
            return (cands[0] if cands else int(rng.integers(0, r))), 1
        for c in cands:
            if pow(g, int(c), p) == x:
                return int(c), t
    raise RuntimeError("no verified logarithm")


# ---------------------------------------------------------------------------------------------------------------
def check_against_qiskit(p=13, g=2, a=7, m=5):
    """Exact outcome distribution of both circuits (state vector) vs the formula above."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate, QFTGate
    from qiskit.quantum_info import Statevector
    from scipy.linalg import expm
    r = p - 1; M = 2 ** m
    nw = int(np.ceil(np.log2(p)))
    x = pow(g, a, p)

    def perm(mult):
        U = np.zeros((2 ** nw, 2 ** nw))
        for y in range(2 ** nw):
            U[(mult * y) % p if 1 <= y < p else y, y] = 1
        return U
    Ug, Ux = perm(g), perm(x)
    tau = TAU
    res = {}
    for kind in ["DT", "CT"]:
        qc = QuantumCircuit(nw + 2 * m)
        w = list(range(nw)); r1 = list(range(nw, nw + m)); r2 = list(range(nw + m, nw + 2 * m))
        qc.x(w[0])
        qc.h(r1 + r2)
        for reg, U in ((r1, Ug), (r2, Ux)):
            for j in range(m):
                if kind == "DT":
                    V = np.linalg.matrix_power(U, 2 ** j)
                else:
                    H = U + U.T
                    V = expm(-1j * tau * (2 ** j) * H)
                qc.append(UnitaryGate(V).control(1), [reg[j]] + w)
            qc.append(QFTGate(m).inverse(), reg)
        pr = Statevector(qc).probabilities()                # little-endian: index = w + 2^nw c1 + 2^(nw+m) c2
        P = pr.reshape(M, M, 2 ** nw).sum(2)                  # [c2, c1]
        # formula (phase convention: controlled-U applied as U^{c}, eigenphase phi -> peak at c = M phi)
        k = np.arange(r)
        if kind == "DT":
            ph1, ph2 = k / r, (k * a % r) / r
        else:
            ph1, ph2 = -(tau / np.pi) * np.cos(2 * np.pi * k / r), -(tau / np.pi) * np.cos(2 * np.pi * (k * a % r) / r)
        cc = np.arange(M)

        def F2(ph):
            d = ph[:, None] - cc[None, :] / M
            den = (M * np.sin(np.pi * d)) ** 2
            v = np.sin(np.pi * M * d) ** 2 / np.where(np.abs(den) < 1e-30, 1, den)
            v[np.abs(np.sin(np.pi * d)) < 1e-12] = 1.0
            return v
        Pf = (F2(ph2)[:, :, None] * F2(ph1)[:, None, :]).sum(0) / r   # [c2, c1]
        res[kind] = float(0.5 * np.abs(P - Pf).sum())
    return res


if __name__ == "__main__":
    # Section 5.1.3: the sampled outcome distribution against a Qiskit state-vector simulation of both full circuits
    for p, g, a in ((11, 2, 7), (13, 2, 7)):
        print(f"p = {p}: total-variation distance, formula vs Qiskit:", check_against_qiskit(p=p, g=g, a=a))
