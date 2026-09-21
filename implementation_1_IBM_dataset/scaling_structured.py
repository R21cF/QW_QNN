"""
Scaling of one walk step: dense synthesis against a structured construction.

Dense synthesis of an n-qubit operator costs O(4^n) two-qubit gates, and
n = log2(N d), so the cost is O(N^2 d^2) in the graph size.  The structured
constructions should be polylogarithmic in N.  This measures both, on the two
families where an arithmetic description of the colour classes exists, and
fits an exponent to each so the claim is a number rather than an impression.
"""

import json

import numpy as np
from scipy.stats import unitary_group

from structured_walk import HypercubeWalk, TorusWalk, cost

DENSE_QUBIT_CAP = 9   # 4^10 synthesis is not worth the wall clock

rows = []

print("ONE WALK STEP -- dense synthesis vs structured construction")
print(f"{'family':<12} {'graph':<10} {'N':>6} {'qubits':>7} "
      f"{'dense depth':>12} {'dense CX':>9} {'str depth':>10} {'str CX':>7} "
      f"{'CX saving':>10}")
print("-" * 92)

for m in range(2, 8):
    w = HypercubeWalk(m)
    c = unitary_group.rvs(w.d, random_state=1)
    ds, cs = cost(w.step_circuit(c))
    if w.n_qubits <= DENSE_QUBIT_CAP:
        dd, cd = cost(w.dense_circuit(c))
        sav = f"{cd / max(cs, 1):9.1f}x"
    else:
        dd = cd = None
        sav = "        --"
    rows.append(dict(family="hypercube", name=f"Q{m}", N=w.n_nodes,
                     qubits=w.n_qubits, dense_depth=dd, dense_cx=cd,
                     str_depth=ds, str_cx=cs))
    print(f"{'hypercube':<12} {'Q' + str(m):<10} {w.n_nodes:>6} {w.n_qubits:>7} "
          f"{(dd if dd else '--'):>12} {(cd if cd else '--'):>9} "
          f"{ds:>10} {cs:>7} {sav:>10}")

for dims in [(8,), (16,), (32,), (64,), (128,), (256,), (512,), (1024,)]:
    w = TorusWalk(dims)
    c = unitary_group.rvs(w.d, random_state=1)
    ds, cs = cost(w.step_circuit(c))
    if w.n_qubits <= DENSE_QUBIT_CAP:
        dd, cd = cost(w.dense_circuit(c))
        sav = f"{cd / max(cs, 1):9.1f}x"
    else:
        dd = cd = None
        sav = "        --"
    rows.append(dict(family="cycle", name=f"C{dims[0]}", N=w.n_nodes,
                     qubits=w.n_qubits, dense_depth=dd, dense_cx=cd,
                     str_depth=ds, str_cx=cs))
    print(f"{'cycle':<12} {'C' + str(dims[0]):<10} {w.n_nodes:>6} {w.n_qubits:>7} "
          f"{(dd if dd else '--'):>12} {(cd if cd else '--'):>9} "
          f"{ds:>10} {cs:>7} {sav:>10}")

for dims in [(4, 4), (8, 4), (8, 8), (16, 8), (16, 16), (32, 16)]:
    w = TorusWalk(dims)
    c = unitary_group.rvs(w.d, random_state=1)
    ds, cs = cost(w.step_circuit(c))
    if w.n_qubits <= DENSE_QUBIT_CAP:
        dd, cd = cost(w.dense_circuit(c))
        sav = f"{cd / max(cs, 1):9.1f}x"
    else:
        dd = cd = None
        sav = "        --"
    nm = f"T{dims[0]}x{dims[1]}"
    rows.append(dict(family="torus", name=nm, N=w.n_nodes, qubits=w.n_qubits,
                     dense_depth=dd, dense_cx=cd, str_depth=ds, str_cx=cs))
    print(f"{'torus':<12} {nm:<10} {w.n_nodes:>6} {w.n_qubits:>7} "
          f"{(dd if dd else '--'):>12} {(cd if cd else '--'):>9} "
          f"{ds:>10} {cs:>7} {sav:>10}")

print("-" * 92)

# ------------------------------------------------------------- exponents --


def fit(rows, family, key):
    sel = [(r["N"], r[key]) for r in rows
           if r["family"] == family and r.get(key)]
    if len(sel) < 3:
        return None
    n = np.log(np.array([s[0] for s in sel], dtype=float))
    y = np.log(np.array([s[1] for s in sel], dtype=float))
    return float(np.polyfit(n, y, 1)[0])


print("\nfitted exponent alpha in  CX  ~  N^alpha")
print(f"{'family':<12} {'dense':>8} {'structured':>12}")
print("-" * 34)
for fam in ("hypercube", "cycle", "torus"):
    a_d, a_s = fit(rows, fam, "dense_cx"), fit(rows, fam, "str_cx")
    print(f"{fam:<12} {(f'{a_d:8.2f}' if a_d else '      --')} "
          f"{(f'{a_s:12.2f}' if a_s else '          --')}")

print("""
Dense synthesis is O(4^n) with n = log2(N d), so alpha near 2 is expected and
is what a fit over the range that could be built should show.  A structured
alpha well below 1 means the cost is polylogarithmic in N: the walk step stops
being the bottleneck.
""")

json.dump(rows, open("scaling_structured.json", "w"), indent=2)
print("wrote scaling_structured.json")
