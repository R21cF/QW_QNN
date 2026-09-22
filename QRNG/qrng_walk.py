"""
Random-number generation from a discrete-time quantum walk on the cycle.

Builds the coined walk on the 2^n-cycle as a Qiskit circuit (n position qubits,
one coin qubit; Hadamard coin; coin-controlled increment/decrement shift),
measures the position register to obtain an n-bit word, and characterises the
resulting source: exact output distribution, min-entropy, total-variation
distance from uniform, and transpiled circuit cost. The H^{otimes n} generator
is included as the reference source.

Writes qrng_results.json and fig_circ_qrng.pdf/.png.
Needs qiskit, qiskit-aer, numpy, matplotlib.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

SURFACE = "#ffffff"; INK = "#0b0b0b"; INK_2 = "#52514e"; GRID = "#e3e2dd"
WALK = "#2a78d6"; REF = "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

BASIS = ["cx", "rz", "sx", "x"]


# ----------------------------------------------------------------- circuit
def increment(n):
    """|x> -> |x+1 mod 2^n> on n qubits, little-endian."""
    qc = QuantumCircuit(n, name="INC")
    for i in range(n - 1, 0, -1):
        qc.mcx(list(range(i)), i)
    qc.x(0)
    return qc


def walk_circuit(n, steps, measure=True, barriers=False):
    """DTQW on the 2^n-cycle: H coin, coin-controlled shift, position read out."""
    coin = QuantumRegister(1, "c")
    pos = QuantumRegister(n, "q")
    qc = QuantumCircuit(coin, pos)
    inc = increment(n).to_gate(label="$S_{+}$").control(1, ctrl_state=1, label="$S_{+}$")
    dec = increment(n).inverse().to_gate(label="$S_{-}$").control(1, ctrl_state=0, label="$S_{-}$")
    qc.h(coin[0])                       # symmetric initial coin state
    qc.s(coin[0])                       # (|0> + i|1>)/sqrt2 -- unbiased walk
    qc.x(pos[n - 1])                    # start at x = 2^(n-1)
    if barriers:
        qc.barrier()
    for _ in range(steps):
        qc.h(coin[0])
        qc.append(inc, [coin[0]] + list(pos))
        qc.append(dec, [coin[0]] + list(pos))
        if barriers:
            qc.barrier()
    if measure:
        cr = ClassicalRegister(n, "m")
        qc.add_register(cr)
        qc.measure(pos, cr)
    return qc


# ------------------------------------------------- exact evolution (numpy)
def exact_distribution(n, steps):
    """Position distribution of the same walk, by direct statevector iteration."""
    N = 2 ** n
    psi = np.zeros((2, N), dtype=complex)
    psi[0, N // 2] = 1 / np.sqrt(2)
    psi[1, N // 2] = 1j / np.sqrt(2)
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for _ in range(steps):
        c = h @ psi
        new = np.zeros_like(psi)
        new[1] = np.roll(c[1], 1)        # coin |1> -> increment
        new[0] = np.roll(c[0], -1)       # coin |0> -> decrement
        psi = new
    return (np.abs(psi) ** 2).sum(axis=0)


def stats(p):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    nz = p[p > 0]
    return {
        "min_entropy_bits": float(-np.log2(p.max())),
        "shannon_entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "tv_from_uniform": float(0.5 * np.abs(p - 1 / len(p)).sum()),
        "max_prob": float(p.max()),
    }


N_QUBITS = 5
STEP_GRID = list(range(1, 65))

exact = {t: exact_distribution(N_QUBITS, t) for t in STEP_GRID}
per_step = {t: stats(exact[t]) for t in STEP_GRID}

# best and worst step counts by min-entropy
best_t = max(STEP_GRID, key=lambda t: per_step[t]["min_entropy_bits"])
worst_t = min(STEP_GRID, key=lambda t: per_step[t]["min_entropy_bits"])

# ------------------------------------------------------------- circuit cost
sim = AerSimulator()
cost = {}
for t in (1, 8, 32):
    qc = walk_circuit(N_QUBITS, t)
    tq = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
    cost[t] = {"cx": tq.count_ops().get("cx", 0), "depth": tq.depth(),
               "qubits": qc.num_qubits}
ref = QuantumCircuit(N_QUBITS, N_QUBITS)
ref.h(range(N_QUBITS)); ref.measure(range(N_QUBITS), range(N_QUBITS))
ref_t = transpile(ref, basis_gates=BASIS, optimization_level=3, seed_transpiler=7)
ref_cost = {"cx": ref_t.count_ops().get("cx", 0), "depth": ref_t.depth(),
            "qubits": N_QUBITS}

# ------------------------------------------- sampled bitstream + basic tests
SHOTS = 200000
T_SAMPLE = best_t
qc = walk_circuit(N_QUBITS, T_SAMPLE)
counts = sim.run(transpile(qc, sim, seed_transpiler=7), shots=SHOTS, seed_simulator=1234
                 ).result().get_counts()
emp = np.zeros(2 ** N_QUBITS)
for bits, c in counts.items():
    emp[int(bits, 2)] = c
emp_p = emp / emp.sum()

# agreement between Aer circuit and the independent numpy evolution
tv_circuit_vs_exact = float(0.5 * np.abs(emp_p - exact[T_SAMPLE]).sum())

# monobit test on the concatenated bitstream
bitstream = "".join(b for b, c in counts.items() for _ in range(c))
ones = bitstream.count("1"); total = len(bitstream)
s_obs = abs(2 * ones - total) / np.sqrt(total)
from math import erfc
monobit_p = erfc(s_obs / np.sqrt(2))

# chi-square uniformity of the 32 words
chi2 = float(((emp - SHOTS / 2 ** N_QUBITS) ** 2 / (SHOTS / 2 ** N_QUBITS)).sum())


# ------------------------------------------------------- per-bit structure
def bit_bias(p, n):
    return [float(sum(p[x] for x in range(2 ** n) if (x >> b) & 1)) for b in range(n)]


def drop_lsb(p, n):
    q = np.zeros(2 ** (n - 1))
    for x in range(2 ** n):
        q[x >> 1] += p[x]
    return q


bias_best = bit_bias(exact[best_t], N_QUBITS)
tail = stats(drop_lsb(exact[best_t], N_QUBITS))

# ------------------------------------------------------------------ figure
from qiskit.visualization import circuit_drawer
fig_qc = walk_circuit(3, 2, measure=True, barriers=True)
fig = circuit_drawer(fig_qc, output="mpl", fold=-1, scale=0.85,
                     style={"name": "bw", "fold": -1})
fig.savefig("fig_circ_qrng.pdf", bbox_inches="tight")
fig.savefig("fig_circ_qrng.png", dpi=200, bbox_inches="tight")
plt.close(fig)

out = {
    "n_position_qubits": N_QUBITS, "cycle_length": 2 ** N_QUBITS,
    "coin": "Hadamard", "initial_coin": "(|0>+i|1>)/sqrt(2)",
    "best_steps_by_min_entropy": best_t, "best": per_step[best_t],
    "worst_steps_by_min_entropy": worst_t, "worst": per_step[worst_t],
    "steps_8": per_step[8], "steps_32": per_step[32],
    "uniform_min_entropy_bits": float(N_QUBITS),
    "circuit_cost": cost, "reference_hadamard_cost": ref_cost,
    "shots": SHOTS, "sampled_at_steps": T_SAMPLE,
    "tv_circuit_vs_exact": tv_circuit_vs_exact,
    "monobit_ones_fraction": ones / total, "monobit_p_value": monobit_p,
    "chi2_uniformity_31df": chi2,
    "empirical_min_entropy_bits": stats(emp_p)["min_entropy_bits"],
    "bit_bias_at_best_t": bias_best,
    "min_entropy_without_lsb_bits": tail["min_entropy_bits"],
    "steps_8_bit_bias": bit_bias(exact[8], N_QUBITS),
}
json.dump(out, open("qrng_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))
