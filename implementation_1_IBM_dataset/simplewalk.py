"""A plain coined quantum walk on a cycle, used as a feature map."""

from __future__ import annotations

import numpy as np

__all__ = ["CycleWalkMap", "walk_circuit"]

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


class CycleWalkMap:
    """Statevector feature map. State held as (2, 2^d): coin, position."""

    def __init__(self, n_features: int, layers: int = 2,
                 coin_angle: float | None = None, scale: float = 1.0):
        self.d = int(n_features)
        self.N = 1 << self.d
        self.n_qubits = self.d + 1
        self.layers = layers
        self.scale = scale
        self.coin = _H if coin_angle is None else _ry(coin_angle)

        self._fwd = (np.arange(self.N) - 1) % self.N   # S: coin 1 -> p+1
        self._bwd = (np.arange(self.N) + 1) % self.N   # S: coin 0 -> p-1

        # bit j of each site, for the product phase layer
        sites = np.arange(self.N)
        self._bits = np.stack([(sites >> j) & 1 for j in range(self.d)])

    def _phases(self, x):
        """exp(i * 2 * lambda * sum_j x_j * bit_j(p)) -- the Z-map phase layer."""
        ang = 2.0 * self.scale * (np.asarray(x, float) @ self._bits)
        return np.exp(1j * ang)

    def state(self, x: np.ndarray) -> np.ndarray:
        psi = np.full((2, self.N), 1.0 / np.sqrt(2 * self.N), dtype=complex)
        ph = self._phases(x)
        for _l in range(self.layers):
            psi = psi * ph
            psi = self.coin @ psi
            psi = np.stack([psi[0][self._bwd], psi[1][self._fwd]])
        return psi.reshape(-1)

    def states(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.state(x) for x in X])


def walk_circuit(fmap: CycleWalkMap, x: np.ndarray):
    """The same map as a Qiskit circuit, for resource accounting."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import XGate, UnitaryGate

    qc_c = QuantumRegister(1, "c")
    qc_p = QuantumRegister(fmap.d, "p")
    qc = QuantumCircuit(qc_c, qc_p, name="walk-fm")
    qc.h(qc_c[0])
    for q in qc_p:
        qc.h(q)

    def ripple(sign, ctrl_state):
        d = fmap.d

        def mcx(k, target):
            ctrls = [qc_c[0]] + list(qc_p[:k])
            cs = ctrl_state | (((1 << k) - 1) << 1)
            qc.append(XGate().control(len(ctrls), ctrl_state=cs),
                      ctrls + [target])

        if sign > 0:
            for k in range(d - 1, 0, -1):
                mcx(k, qc_p[k])
            mcx(0, qc_p[0])
        else:
            mcx(0, qc_p[0])
            for k in range(1, d):
                mcx(k, qc_p[k])

    for _l in range(fmap.layers):
        for j in range(fmap.d):
            qc.p(2.0 * fmap.scale * float(x[j]), qc_p[j])
        qc.append(UnitaryGate(fmap.coin, label="C"), [qc_c[0]])
        ripple(-1, 0)
        ripple(+1, 1)
    return qc
