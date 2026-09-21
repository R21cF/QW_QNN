
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import XGate

D = 4          # features, hence position qubits
STYLE = {"name": "bw", "fold": -1}


def ripple(qc, coin, pos, sign, ctrl_state):
    d = len(pos)

    def mcx(k, target):
        ctrls = [coin[0]] + list(pos[:k])
        cs = ctrl_state | (((1 << k) - 1) << 1)
        qc.append(XGate().control(len(ctrls), ctrl_state=cs), ctrls + [target])

    if sign > 0:
        for k in range(d - 1, 0, -1):
            mcx(k, pos[k])
        mcx(0, pos[0])
    else:
        mcx(0, pos[0])
        for k in range(1, d):
            mcx(k, pos[k])


def walk_layer(qc, coin, pos, xs, boxed=True):
    for j, xj in enumerate(xs):
        qc.p(2 * xj, pos[j])
    qc.h(coin[0])
    if boxed:
        sub = QuantumCircuit(1, len(pos), name="")
    ripple(qc, coin, pos, -1, 0)
    ripple(qc, coin, pos, +1, 1)


def save(qc, fname, title, scale=0.85):
    fig = qc.draw("mpl", style=STYLE, fold=-1, scale=scale)
    fig.suptitle(title, fontsize=9, y=1.02)
    fig.savefig(fname + ".pdf", bbox_inches="tight")
    fig.savefig(fname + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname + ".pdf")


# ---------------------------------------------------------- 1. feature map
x = ParameterVector("x", D)
coin = QuantumRegister(1, "c")
pos = QuantumRegister(D, "p")
qc = QuantumCircuit(coin, pos)
qc.h(coin[0])
for q in pos:
    qc.h(q)
qc.barrier(label="init")
for j in range(D):
    qc.p(2 * x[j], pos[j])
qc.barrier(label="D(x)")
qc.h(coin[0])
qc.barrier(label="coin")
ripple(qc, coin, pos, -1, 0)
ripple(qc, coin, pos, +1, 1)
qc.barrier(label="shift S")
save(qc, "fig_circ_walk_fm",
     f"Walk feature map, one layer, {D} features "
     f"({D + 1} qubits): H init, phase encoding, Hadamard coin, shift")

# ------------------------------------------------------- 2. shift detail
coin = QuantumRegister(1, "c")
pos = QuantumRegister(D, "p")
qc = QuantumCircuit(coin, pos)
ripple(qc, coin, pos, -1, 0)
qc.barrier(label="p-1")
ripple(qc, coin, pos, +1, 1)
qc.barrier(label="p+1")
save(qc, "fig_circ_walk_shift",
     "Shift operator: two coin-controlled ripple increments modulo $2^d$. "
     "The discarded top carry is the cycle's wraparound.")

# ------------------------------------------------------------- 3. kernel
xa = ParameterVector("x", D)
ya = ParameterVector("y", D)
coin = QuantumRegister(1, "c")
pos = QuantumRegister(D, "p")
qc = QuantumCircuit(coin, pos)
qc.h(coin[0])
for q in pos:
    qc.h(q)
for j in range(D):
    qc.p(2 * xa[j], pos[j])
qc.h(coin[0])
ripple(qc, coin, pos, -1, 0)
ripple(qc, coin, pos, +1, 1)
qc.barrier(label="U(x)")
inv = QuantumCircuit(coin, pos)
for j in range(D):
    inv.p(2 * ya[j], pos[j])
inv.h(coin[0])
ripple(inv, coin, pos, -1, 0)
ripple(inv, coin, pos, +1, 1)
qc.compose(inv.inverse(), inplace=True)
qc.barrier(label="U*(y)")
qc.measure_all()
save(qc, "fig_circ_kernel",
     "Fidelity kernel overlap circuit. "
     "$K(x,y)=|\\langle\\Phi(y)|\\Phi(x)\\rangle|^2$ is the probability of "
     "the all-zero outcome.", scale=0.72)

# ---------------------------------------------------------------- 4. VQC
theta = ParameterVector("θ", 2 * (D + 1) * 1 + (D + 1))
coin = QuantumRegister(1, "c")
pos = QuantumRegister(D, "p")
qc = QuantumCircuit(coin, pos)
qc.h(coin[0])
for q in pos:
    qc.h(q)
for j in range(D):
    qc.p(2 * x[j], pos[j])
qc.h(coin[0])
ripple(qc, coin, pos, -1, 0)
ripple(qc, coin, pos, +1, 1)
qc.barrier(label="walk map")

allq = list(coin) + list(pos)
n = len(allq)
k = 0
for q in range(n):
    qc.ry(theta[k], allq[q]); k += 1
for q in range(n):
    qc.rz(theta[k], allq[q]); k += 1
for q in range(n):
    qc.cx(allq[q], allq[(q + 1) % n])
qc.barrier(label="ansatz rep")
for q in range(n):
    qc.ry(theta[k], allq[q]); k += 1
qc.barrier(label="final RY")
qc.measure(coin[0], 0) if qc.num_clbits else None
save(qc, "fig_circ_vqc",
     "Variational classifier: walk feature map, hardware-efficient ansatz "
     "(one of three repetitions shown), read out as $\\langle Z\\rangle$ "
     "on the coin qubit.", scale=0.72)

print("done")
