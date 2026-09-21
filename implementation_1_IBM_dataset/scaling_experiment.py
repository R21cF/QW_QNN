"""Does the coloured construction actually scale better, and is it the same walk?"""

import numpy as np
import networkx as nx
from scipy.stats import unitary_group
from qiskit.quantum_info import Operator

from efficient_walk import ColouredWalk, cost, per_vertex_coin_circuit

RNG = np.random.default_rng(17)


# ------------------------------------------------------------ correctness --

print("[1] the efficient circuit implements the same operator as the dense one")
ok = True
for name, g in [("cycle-8", nx.cycle_graph(8)),
                ("grid 2x4", nx.grid_2d_graph(2, 4)),
                ("grid 3x4", nx.grid_2d_graph(3, 4)),
                ("petersen", nx.petersen_graph()),
                ("Q4", nx.hypercube_graph(4))]:
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    w = ColouredWalk(g)
    coin = unitary_group.rvs(w.d, random_state=int(RNG.integers(1 << 30)))

    u_eff = Operator(w.step_circuit_efficient(coin)).data
    u_ref = w.step_matrix(coin)
    err = np.abs(u_eff - u_ref).max()
    good = err < 1e-9
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<10} "
          f"Delta={max(1, w.max_degree)} colours={w.k} "
          f"qubits={w.n_qubits}  max|delta| = {err:.2e}")

if not ok:
    raise SystemExit("efficient construction does not match -- stop here")


# --------------------------------------------------------------- scaling --

print("\n[2] cost of ONE walk step, uniform coin")
print(f"{'graph':<16} {'N':>4} {'Delta':>5} {'col':>4} {'qb':>3} "
      f"{'dense depth':>12} {'dense CX':>9} {'eff depth':>10} {'eff CX':>7} "
      f"{'CX ratio':>9}")
print("-" * 92)

GRAPHS = [
    ("cycle-8", nx.cycle_graph(8)),
    ("grid 2x4", nx.grid_2d_graph(2, 4)),
    ("cycle-16", nx.cycle_graph(16)),
    ("grid 3x4", nx.grid_2d_graph(3, 4)),
    ("grid 4x4", nx.grid_2d_graph(4, 4)),
    ("Q4", nx.hypercube_graph(4)),
    ("grid 4x6", nx.grid_2d_graph(4, 6)),
    ("grid 4x8", nx.grid_2d_graph(4, 8)),
    ("Q5", nx.hypercube_graph(5)),
    ("grid 6x8", nx.grid_2d_graph(6, 8)),
    ("grid 8x8", nx.grid_2d_graph(8, 8)),
]

rows = []
for name, g in GRAPHS:
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    w = ColouredWalk(g)
    coin = unitary_group.rvs(w.d, random_state=3)

    d_eff, cx_eff = cost(w.step_circuit_efficient(coin))

    # the dense operator is 4^n -- stop building it once that is pointless
    if w.n_qubits <= 9:
        d_den, cx_den = cost(w.step_circuit_dense(coin))
        ratio = f"{cx_den / max(cx_eff, 1):7.1f}x"
        dstr, cstr = f"{d_den:12d}", f"{cx_den:9d}"
    else:
        d_den = cx_den = None
        ratio = "      --"
        dstr, cstr = f"{'not built':>12}", f"{'--':>9}"

    rows.append((name, w.n_nodes, w.max_degree, w.k, w.n_qubits,
                 d_den, cx_den, d_eff, cx_eff))
    print(f"{name:<16} {w.n_nodes:>4} {w.max_degree:>5} {w.k:>4} {w.n_qubits:>3} "
          f"{dstr} {cstr} {d_eff:>10} {cx_eff:>7} {ratio:>9}")


print("\n[3] cost of a per-vertex (structure-dependent) coin")
print(f"{'graph':<16} {'N':>4} {'qb':>3} {'uniform CX':>11} {'per-vertex CX':>14} "
      f"{'penalty':>9}")
print("-" * 62)

for name, g in GRAPHS[:7]:
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    w = ColouredWalk(g)
    coin = unitary_group.rvs(w.d, random_state=3)
    _d, cx_uniform = cost(w.step_circuit_efficient(coin))

    coins = [unitary_group.rvs(w.d, random_state=100 + v) for v in range(w.n_nodes)]
    qc = per_vertex_coin_circuit(w, coins)
    w.shift_circuit(qc, qc.qregs[0], qc.qregs[1])
    _d2, cx_pv = cost(qc)

    print(f"{name:<16} {w.n_nodes:>4} {w.n_qubits:>3} {cx_uniform:>11} "
          f"{cx_pv:>14} {cx_pv / max(cx_uniform, 1):>8.1f}x")

np.save("scaling_rows.npy", np.array(
    [(n, N, D, k, q, (dd if dd else -1), (cd if cd else -1), de, ce)
     for (n, N, D, k, q, dd, cd, de, ce) in rows],
    dtype=object), allow_pickle=True)
print("\nwrote scaling_rows.npy")
