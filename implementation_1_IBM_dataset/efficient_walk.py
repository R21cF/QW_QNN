"""
A walk-step circuit that does not synthesise a dense operator.

The naive construction appends each step S(I (x) C) as one dense UnitaryGate on
all n qubits, and generic n-qubit synthesis costs O(4^n) two-qubit gates.  On
the eight-vertex grid that is ~420 CX per step, which is why the classifier of
Chapter 4 transpiles to depth 6119.

The construction here follows Douglas and Wang: proper-edge-colour the graph,
index the coin by COLOUR rather than by a per-vertex neighbour ordering, and
the shift decomposes as

    S  =  sum_c  |c><c| (x) M_c

where M_c is the involution that swaps the endpoints of every edge in colour
class c.  Because a colour class is a matching, M_c is a permutation of the
position register alone, and the coin index is untouched by the shift.  Each
term is therefore a small permutation on ceil(log2 N) qubits, controlled on
the coin register -- never a generic operator on all n qubits.

By Vizing's theorem a graph needs at most Delta+1 colours and a bipartite one
exactly Delta, so the number of terms grows with the DEGREE, not with the
number of vertices.  That is where the scaling comes from.

The cost of the coin is a separate matter, and the more consequential one for
this thesis.  A coin that is the same at every vertex is a single gate on the
coin register: free.  A coin that varies per vertex -- axis 1 -- is a distinct
controlled operation for each of the N vertices, and no colouring helps with
that.  ``StepCost`` measures both so the trade is visible rather than implied.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate

BASIS = ["cx", "rz", "sx", "x"]


# --------------------------------------------------------------------------
# edge colouring
# --------------------------------------------------------------------------


def edge_colouring(g: nx.Graph) -> dict[tuple[int, int], int]:
    """Proper edge colouring by greedy colouring of the line graph.

    Greedy is not guaranteed to hit Vizing's bound, but it is within one or
    two colours in practice and every colour class it returns is a genuine
    matching, which is all the construction requires.  ``n_colours`` is
    reported alongside the maximum degree so the gap is visible.
    """
    lg = nx.line_graph(g)
    colours = nx.greedy_color(lg, strategy="largest_first")
    return {tuple(sorted(e)): c for e, c in colours.items()}


def colour_classes(g: nx.Graph) -> list[list[tuple[int, int]]]:
    col = edge_colouring(g)
    k = max(col.values()) + 1 if col else 0
    classes = [[] for _ in range(k)]
    for e, c in col.items():
        classes[c].append(e)
    return classes


# --------------------------------------------------------------------------
# the walk, defined by its colouring
# --------------------------------------------------------------------------


class ColouredWalk:
    """DTQW whose coin directions are edge colours.

    S|v, c> = |m_c(v), c>, with m_c(v) = v when v has no edge of colour c.
    The coin index is preserved by the shift, unlike the flip-flop convention
    used for the dense construction; both are legitimate coined walks, and
    the coloured one is the one that admits a cheap circuit.
    """

    def __init__(self, graph: nx.Graph):
        self.g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        self.n_nodes = self.g.number_of_nodes()
        self.classes = colour_classes(self.g)
        self.k = len(self.classes)
        self.max_degree = max((d for _v, d in self.g.degree()), default=0)

        self.d = 1 << max(1, int(self.k - 1).bit_length())
        self.n_coin_q = int(np.log2(self.d))
        self.n_pos_q = max(1, int(np.ceil(np.log2(self.n_nodes))))
        self.n_qubits = self.n_pos_q + self.n_coin_q
        self.n_pad = 1 << self.n_pos_q
        self.dim = self.n_pad * self.d

        self.matchings = []
        for cls in self.classes:
            m = np.arange(self.n_pad)
            for a, b in cls:
                m[a], m[b] = b, a
            self.matchings.append(m)

    # -- matrices (reference) --------------------------------------------

    def matching_matrix(self, c: int) -> np.ndarray:
        p = np.zeros((self.n_pad, self.n_pad))
        p[self.matchings[c], np.arange(self.n_pad)] = 1.0
        return p

    def shift_matrix(self) -> np.ndarray:
        s = np.zeros((self.dim, self.dim), dtype=complex)
        for c in range(self.d):
            m = self.matching_matrix(c) if c < self.k else np.eye(self.n_pad)
            for v in range(self.n_pad):
                s[int(np.argmax(m[:, v])) * self.d + c, v * self.d + c] = 1.0
        return s

    def step_matrix(self, coin: np.ndarray) -> np.ndarray:
        big = np.zeros((self.dim, self.dim), dtype=complex)
        for v in range(self.n_pad):
            big[v * self.d:(v + 1) * self.d, v * self.d:(v + 1) * self.d] = coin
        return self.shift_matrix() @ big

    # -- circuits ---------------------------------------------------------

    def registers(self):
        return QuantumRegister(self.n_coin_q, "c"), QuantumRegister(self.n_pos_q, "p")

    def shift_circuit(self, qc, qc_coin, qc_pos):
        """sum_c |c><c| (x) M_c, as k controlled position permutations."""
        for c in range(self.k):
            perm = self.matchings[c]
            if np.array_equal(perm, np.arange(self.n_pad)):
                continue
            mat = np.zeros((self.n_pad, self.n_pad), dtype=complex)
            mat[perm, np.arange(self.n_pad)] = 1.0
            gate = UnitaryGate(mat, label=f"M{c}").control(
                self.n_coin_q, ctrl_state=c)
            qc.append(gate, list(qc_coin) + list(qc_pos))
        return qc

    def step_circuit_efficient(self, coin: np.ndarray) -> QuantumCircuit:
        """Uniform coin on the coin register, then the coloured shift."""
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, qc_pos, name="step-eff")
        qc.append(UnitaryGate(coin, label="C"), list(qc_coin))
        self.shift_circuit(qc, qc_coin, qc_pos)
        return qc

    def step_circuit_dense(self, coin: np.ndarray) -> QuantumCircuit:
        """The naive construction, for comparison: one dense n-qubit gate."""
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, qc_pos, name="step-dense")
        qc.append(UnitaryGate(self.step_matrix(coin), label="U"), list(qc.qubits))
        return qc

    def data_phase_circuit(self, x: np.ndarray, scale: float = 1.0
                           ) -> QuantumCircuit:
        """Encode data as diag(exp(i * scale * x_v)) on the position register.

        A diagonal unitary on m qubits costs 2^m - 1 CX under the standard
        Walsh-Hadamard (Gray-code) decomposition, independent of how many
        distinct values the data takes -- and 2^m = N, so this is one CX per
        vertex rather than one multi-controlled gate per vertex.  It is how
        the data enters without reintroducing a per-vertex coin.
        """
        from qiskit.circuit.library import Diagonal
        ang = np.zeros(self.n_pad)
        ang[: len(x)] = scale * np.asarray(x, dtype=float)
        qc_coin, qc_pos = self.registers()
        qc = QuantumCircuit(qc_coin, qc_pos, name="D(x)")
        qc.append(Diagonal(np.exp(1j * ang)), list(qc_pos))
        return qc


# --------------------------------------------------------------------------
# cost measurement
# --------------------------------------------------------------------------


def cost(qc: QuantumCircuit, opt: int = 3, seed: int = 7):
    t = transpile(qc, basis_gates=BASIS, optimization_level=opt,
                  seed_transpiler=seed)
    return t.depth(), t.count_ops().get("cx", 0)


def per_vertex_coin_circuit(w: ColouredWalk, coins: list[np.ndarray]
                            ) -> QuantumCircuit:
    """Axis 1 priced honestly: a different coin at every vertex.

    Each vertex needs its coin applied conditioned on the position register
    holding that vertex, so the gate count grows with N however the shift is
    built.  This is the cost of structure-dependent coins, and no edge
    colouring reduces it.
    """
    qc_coin, qc_pos = w.registers()
    qc = QuantumCircuit(qc_coin, qc_pos, name="C(v)")
    for v in range(w.n_nodes):
        g = UnitaryGate(coins[v], label=f"C{v}").control(
            w.n_pos_q, ctrl_state=v)
        qc.append(g, list(qc_pos) + list(qc_coin))
    return qc
