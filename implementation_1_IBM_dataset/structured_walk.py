"""Walk-step circuits built from structure, not from synthesis."""

from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate, XGate

BASIS = ["cx", "rz", "sx", "x"]


def cost(qc, opt=3, seed=7):
    t = transpile(qc, basis_gates=BASIS, optimization_level=opt,
                  seed_transpiler=seed)
    return t.depth(), t.count_ops().get("cx", 0)


# --------------------------------------------------------------------------
# hypercube
# --------------------------------------------------------------------------


class HypercubeWalk:
    """DTQW on Q_m. Coin dimension m, one direction per dimension."""

    def __init__(self, m: int):
        self.m = m
        self.n_nodes = 1 << m
        self.n_pos_q = m
        self.n_coin_q = max(1, int(np.ceil(np.log2(m))))
        self.d = 1 << self.n_coin_q
        self.n_qubits = self.n_pos_q + self.n_coin_q
        self.dim = (1 << self.n_pos_q) * self.d
        self.graph = nx.hypercube_graph(m)

    def registers(self):
        return (QuantumRegister(self.n_coin_q, "c"),
                QuantumRegister(self.n_pos_q, "p"))

    def step_matrix(self, coin):
        n_pad = 1 << self.n_pos_q
        big = np.zeros((self.dim, self.dim), dtype=complex)
        for v in range(n_pad):
            big[v * self.d:(v + 1) * self.d, v * self.d:(v + 1) * self.d] = coin
        s = np.zeros((self.dim, self.dim), dtype=complex)
        for v in range(n_pad):
            for c in range(self.d):
                u = v ^ (1 << c) if c < self.m else v
                s[u * self.d + c, v * self.d + c] = 1.0
        return s @ big

    def step_circuit(self, coin):
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, qc_pos, name="Q-step")
        qc.append(UnitaryGate(coin, label="C"), list(qc_coin))
        for i in range(self.m):
            qc.append(XGate().control(self.n_coin_q, ctrl_state=i),
                      list(qc_coin) + [qc_pos[i]])
        return qc

    def dense_circuit(self, coin):
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, qc_pos, name="Q-dense")
        qc.append(UnitaryGate(self.step_matrix(coin), label="U"), list(qc.qubits))
        return qc


# --------------------------------------------------------------------------
# cycle and torus
# --------------------------------------------------------------------------


def _increment(qc, reg, ctrl_qubits=None, ctrl_state=None, sign=+1):
    """In-place +/-1 modulo 2^m on ``reg``, optionally controlled."""
    m = len(reg)
    ctrl_qubits = list(ctrl_qubits or [])
    nc = len(ctrl_qubits)

    def mcx(targets_ctrl, target):
        ctrls = ctrl_qubits + list(targets_ctrl)
        if not ctrls:
            qc.x(target)
        else:
            g = XGate().control(len(ctrls),
                                ctrl_state=(ctrl_state if nc else None)
                                if nc == len(ctrls) else None)
            # build the control state: outer controls use ctrl_state, the
            # ripple controls are all |1>
            cs = None
            if nc:
                cs = (ctrl_state & ((1 << nc) - 1)) | \
                     (((1 << len(targets_ctrl)) - 1) << nc)
            g = XGate().control(len(ctrls), ctrl_state=cs)
            qc.append(g, ctrls + [target])

    order = range(m - 1, 0, -1) if sign > 0 else range(1, m)
    if sign > 0:
        for k in order:
            mcx(list(reg[:k]), reg[k])
        mcx([], reg[0])
    else:
        mcx([], reg[0])
        for k in order:
            mcx(list(reg[:k]), reg[k])
    return qc


class TorusWalk:
    """DTQW on a product of cycles, each of length a power of two."""

    def __init__(self, dims):
        self.dims = list(dims)
        self.r = len(dims)
        self.mbits = [int(np.log2(L)) for L in dims]
        for L, b in zip(dims, self.mbits):
            assert 1 << b == L, "each cycle length must be a power of two"
        self.n_pos_q = sum(self.mbits)
        self.n_nodes = 1 << self.n_pos_q
        self.n_dirs = 2 * self.r
        self.n_coin_q = max(1, int(np.ceil(np.log2(self.n_dirs))))
        self.d = 1 << self.n_coin_q
        self.n_qubits = self.n_pos_q + self.n_coin_q
        self.dim = self.n_nodes * self.d

    def registers(self):
        coin = QuantumRegister(self.n_coin_q, "c")
        pos = [QuantumRegister(b, f"p{j}") for j, b in enumerate(self.mbits)]
        return coin, pos

    def _decode(self, v):
        out, rest = [], v
        for b in self.mbits:
            out.append(rest & ((1 << b) - 1))
            rest >>= b
        return out

    def _encode(self, coords):
        v, sh = 0, 0
        for c, b in zip(coords, self.mbits):
            v |= (c & ((1 << b) - 1)) << sh
            sh += b
        return v

    def step_matrix(self, coin):
        big = np.zeros((self.dim, self.dim), dtype=complex)
        for v in range(self.n_nodes):
            big[v * self.d:(v + 1) * self.d, v * self.d:(v + 1) * self.d] = coin
        s = np.zeros((self.dim, self.dim), dtype=complex)
        for v in range(self.n_nodes):
            for c in range(self.d):
                if c < self.n_dirs:
                    j, sgn = c // 2, (1 if c % 2 == 0 else -1)
                    co = self._decode(v)
                    co[j] = (co[j] + sgn) % self.dims[j]
                    u = self._encode(co)
                else:
                    u = v
                s[u * self.d + c, v * self.d + c] = 1.0
        return s @ big

    def step_circuit(self, coin):
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, *qc_pos, name="T-step")
        qc.append(UnitaryGate(coin, label="C"), list(qc_coin))
        for c in range(self.n_dirs):
            j, sgn = c // 2, (1 if c % 2 == 0 else -1)
            _increment(qc, list(qc_pos[j]), ctrl_qubits=list(qc_coin),
                       ctrl_state=c, sign=sgn)
        return qc

    def dense_circuit(self, coin):
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, *qc_pos, name="T-dense")
        qc.append(UnitaryGate(self.step_matrix(coin), label="U"),
                  list(qc.qubits))
        return qc
