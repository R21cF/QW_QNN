"""Order finding as a quantum walk on the functional graph of x -> a x (mod N)."""

from __future__ import annotations

import json
from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit_aer import AerSimulator

SEED = 7
SHOTS = 4096


def order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def functional_graph_cycles(a, N):
    """Cycles of x -> a x mod N restricted to the units of Z_N."""
    seen, cycles = set(), []
    for x in range(1, N):
        if gcd(x, N) != 1 or x in seen:
            continue
        cyc, y = [], x
        while y not in seen:
            seen.add(y)
            cyc.append(y)
            y = (a * y) % N
        cycles.append(cyc)
    return cycles


def walk_unitary(a, N, n):
    """Permutation matrix of x -> a x mod N on 2^n basis states (identity on x >= N)."""
    dim = 2 ** n
    U = np.zeros((dim, dim))
    for x in range(dim):
        y = (a * x) % N if x < N else x
        U[y, x] = 1.0
    return U


def eigenphases_on_cycle(a, N, n):
    """
    Exact eigenphases of the walk restricted to the cycle through 1, as
    fractions of 2 pi.
    """
    U = walk_unitary(a, N, n)
    cyc = [c for c in functional_graph_cycles(a, N) if 1 in c][0]
    sub = U[np.ix_(cyc, cyc)]
    lam = np.linalg.eigvals(sub)
    ph = np.sort(np.mod(np.angle(lam) / (2 * np.pi), 1.0))
    return ph, len(cyc)


def return_probability(a, N, n, tmax):
    U = walk_unitary(a, N, n)
    psi = np.zeros(2 ** n); psi[1] = 1.0
    out = []
    for t in range(tmax + 1):
        out.append(float(abs(psi[1]) ** 2))
        psi = U @ psi
    return out


def qpe_circuit(a, N, n, m):
    """m counting qubits, n target qubits; controlled-U_a^(2^j) as dense gates."""
    U = walk_unitary(a, N, n)
    cnt = QuantumRegister(m, "c")
    tgt = QuantumRegister(n, "x")
    cl = ClassicalRegister(m, "k")
    qc = QuantumCircuit(cnt, tgt, cl)
    qc.h(cnt)
    qc.x(tgt[0])                       # |x> = |1>
    Upow = U.copy()
    for j in range(m):
        g = UnitaryGate(Upow, label=f"U^{2**j}").control(1)
        qc.append(g, [cnt[j], *tgt])
        Upow = Upow @ Upow
    qc.append(QFT(m, inverse=True, do_swaps=True).to_gate(label="QFT^-1"), cnt)
    qc.measure(cnt, cl)
    return qc


def postprocess(counts, a, N, m):
    """Continued fractions on each measured phase; returns per-outcome results."""
    r_true = order(a, N)
    hits, rows = 0, []
    for bits, c in counts.items():
        k = int(bits, 2)
        phase = k / 2 ** m
        frac = Fraction(phase).limit_denominator(N)
        r_guess = frac.denominator
        ok = r_guess > 0 and pow(a, r_guess, N) == 1
        factor = None
        if ok and r_guess % 2 == 0:
            p = gcd(pow(a, r_guess // 2, N) - 1, N)
            q = gcd(pow(a, r_guess // 2, N) + 1, N)
            if 1 < p < N:
                factor = (p, N // p)
            elif 1 < q < N:
                factor = (q, N // q)
        rows.append({"k": k, "phase": phase, "r_guess": r_guess, "count": c,
                     "valid_order": ok, "factor": factor})
        if ok and r_guess == r_true:
            hits += c
    return hits / sum(counts.values()), rows


if __name__ == "__main__":
    sim = AerSimulator(seed_simulator=SEED)
    results = []
    cases = [(15, 7), (15, 2), (15, 11), (15, 4), (21, 2), (21, 5), (21, 8), (33, 2), (35, 3)]
    for N, a in cases:
        n = int(np.ceil(np.log2(N)))
        r = order(a, N)
        cycles = functional_graph_cycles(a, N)
        ph, L = eigenphases_on_cycle(a, N, n)
        assert L == r and np.allclose(ph, np.arange(r) / r), "eigenphases are not k/r"
        ret = return_probability(a, N, n, 2 * r + 1)
        revivals = [t for t, p in enumerate(ret) if p > 0.5]
        assert revivals[:3] == [0, r, 2 * r]
        row = {"N": N, "a": a, "n": n, "r": r,
               "cycle_lengths": sorted(len(c) for c in cycles),
               "eigenphases": [float(x) for x in ph],
               "return_prob": ret, "qpe": {}}
        for m in [2 * n, 2 * n + 2]:
            qc = qpe_circuit(a, N, n, m)
            tq = transpile(qc, sim, optimization_level=1)
            counts = sim.run(tq, shots=SHOTS).result().get_counts()
            p_r, rows = postprocess(counts, a, N, m)
            factors = {str(x["factor"]) for x in rows if x["factor"]}
            tq3 = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3)
            row["qpe"][m] = {"p_correct_r": p_r, "factors_found": sorted(factors),
                             "qubits": m + n, "depth": tq3.depth(),
                             "cx": tq3.count_ops().get("cx", 0),
                             "outcomes": sorted(rows, key=lambda x: -x["count"])[:8]}
            print(f"N={N:2d} a={a:2d} r={r:2d} m={m}: P(r correct)={p_r:.3f} "
                  f"factors={sorted(factors)} qubits={m+n} depth={tq3.depth()} cx={row['qpe'][m]['cx']}")
        results.append(row)
        print(f"   cycles on units: {row['cycle_lengths']}   eigenphases: {np.round(ph,3)}")
    json.dump(results, open("shor_results.json", "w"), indent=1)
    # circuit figure source for N=15, a=7, m=4 (drawn by plot script)
    qc = qpe_circuit(7, 15, 4, 4)
    open("shor_circuit_15_7.txt", "w").write(qc.draw(output="text", fold=120).single_string())
