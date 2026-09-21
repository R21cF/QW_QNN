"""
Two things the raw scaling table does not settle.

First, the fitted exponent over the whole range is contaminated by the
smallest instances, where constant overheads dominate.  Refitting on the
upper half gives the asymptotic behaviour, and the two fits are reported
together so the difference is visible.

Second, and more consequential for this thesis: the structured construction
makes the SHIFT cheap.  It does nothing for the coin.  A coin that varies
per vertex -- axis 1 of the contribution -- needs one controlled operation
per vertex however the shift is built, so its cost is linear in N no matter
what.  If that term dominates, the structured shift buys nothing for the
model actually proposed, and the thesis needs to say so.
"""

import json

import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from structured_walk import TorusWalk, HypercubeWalk, cost

rows = json.load(open("scaling_structured.json"))


def fit(sel_rows, key):
    sel = [(r["N"], r[key]) for r in sel_rows if r.get(key)]
    if len(sel) < 3:
        return None
    n = np.log([s[0] for s in sel]); y = np.log([s[1] for s in sel])
    return float(np.polyfit(n, y, 1)[0])


def fit_log(sel_rows, key):
    """CX ~ a (log2 N)^beta -- the fit that should hold if cost is polylog."""
    sel = [(r["N"], r[key]) for r in sel_rows if r.get(key)]
    if len(sel) < 3:
        return None
    n = np.log(np.log2([s[0] for s in sel])); y = np.log([s[1] for s in sel])
    return float(np.polyfit(n, y, 1)[0])


print("Exponents for the structured construction")
print(f"{'family':<12} {'alpha (all N)':>14} {'alpha (large N)':>16} "
      f"{'beta in (log N)^beta':>22}")
print("-" * 68)
for fam in ("hypercube", "cycle", "torus"):
    fr = [r for r in rows if r["family"] == fam]
    big = fr[len(fr) // 2:]
    a_all, a_big, b = fit(fr, "str_cx"), fit(big, "str_cx"), fit_log(fr, "str_cx")
    print(f"{fam:<12} {a_all:>14.2f} {a_big:>16.2f} {b:>22.2f}")

print("""
alpha falls sharply once the smallest instances are dropped, and a power law
in log2 N fits with a modest exponent -- the structured step is polylogarithmic
in the number of vertices, against the dense step's ~N^2.
""")

# ------------------------------------------------ the per-vertex coin -----

print("Cost of the coin: uniform vs per-vertex (axis 1), structured shift")
print(f"{'graph':<10} {'N':>5} {'qb':>3} {'shift CX':>9} {'uniform CX':>11} "
      f"{'per-vertex CX':>14} {'total penalty':>14}")
print("-" * 72)

pv_rows = []
for dims in [(8,), (16,), (32,), (4, 4), (8, 4), (8, 8)]:
    w = TorusWalk(dims)
    coin = unitary_group.rvs(w.d, random_state=1)

    qc_coin, qc_pos = w.registers()
    shift_only = QuantumCircuit(qc_coin, *qc_pos)
    for c in range(w.n_dirs):
        from structured_walk import _increment
        j, sgn = c // 2, (1 if c % 2 == 0 else -1)
        _increment(shift_only, list(qc_pos[j]), ctrl_qubits=list(qc_coin),
                   ctrl_state=c, sign=sgn)
    _d, cx_shift = cost(shift_only)
    _d, cx_uniform = cost(w.step_circuit(coin))

    # per-vertex coin: one controlled coin per vertex, then the same shift
    qc_coin2, qc_pos2 = w.registers()
    qc = QuantumCircuit(qc_coin2, *qc_pos2)
    allpos = [q for r in qc_pos2 for q in r]
    for v in range(w.n_nodes):
        cv = unitary_group.rvs(w.d, random_state=500 + v)
        qc.append(UnitaryGate(cv, label=f"C{v}").control(
            w.n_pos_q, ctrl_state=v), allpos + list(qc_coin2))
    for c in range(w.n_dirs):
        from structured_walk import _increment
        j, sgn = c // 2, (1 if c % 2 == 0 else -1)
        _increment(qc, list(qc_pos2[j]), ctrl_qubits=list(qc_coin2),
                   ctrl_state=c, sign=sgn)
    _d, cx_pv = cost(qc)

    nm = "C" + str(dims[0]) if len(dims) == 1 else f"T{dims[0]}x{dims[1]}"
    pv_rows.append(dict(name=nm, N=w.n_nodes, qubits=w.n_qubits,
                        cx_shift=cx_shift, cx_uniform=cx_uniform, cx_pv=cx_pv))
    print(f"{nm:<10} {w.n_nodes:>5} {w.n_qubits:>3} {cx_shift:>9} "
          f"{cx_uniform:>11} {cx_pv:>14} {cx_pv / max(cx_uniform, 1):>13.1f}x")

a_pv = fit([{**r, "str_cx": r["cx_pv"], "family": "x"} for r in pv_rows], "str_cx")
a_un = fit([{**r, "str_cx": r["cx_uniform"], "family": "x"} for r in pv_rows], "str_cx")
print(f"\nfitted alpha: uniform coin {a_un:.2f}, per-vertex coin {a_pv:.2f}")
print("""
The per-vertex coin is the dominant term and it grows at least linearly in N.
The structured shift removes the shift from the cost budget; it does not make
a structure-dependent coin affordable.  Axis 1 as implemented in Chapter 4 is
therefore simulation-only regardless of how the shift is built, and a
hardware-realisable walk model has to either share coins across vertices or
encode the per-vertex data somewhere cheaper -- a diagonal phase on the
position register costs N-1 CX once, rather than N controlled coins per step.
""")

json.dump(pv_rows, open("coin_cost.json", "w"), indent=2)
print("wrote coin_cost.json")
