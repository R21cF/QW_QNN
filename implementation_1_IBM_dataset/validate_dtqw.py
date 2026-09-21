"""Validation for dtqw.py."""

import numpy as np
import networkx as nx

from dtqw import (
    DTQW,
    coin_grover,
    coin_hadamard,
    coin_fourier,
    coin_householder,
    coin_parametric,
    fixed,
    structure_dependent,
    step_dependent,
)

RNG = np.random.default_rng(20260919)
ok = lambda name, cond: print(("  PASS  " if cond else "  FAIL  ") + name) or bool(cond)
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(("  PASS  " if cond else "  FAIL  ") + name)


def unitary(m, tol=1e-10):
    return np.allclose(m.conj().T @ m, np.eye(m.shape[0]), atol=tol)


# ---------------------------------------------------------------- coins ----
print("\n[1] coin families are unitary, and real/complex is as advertised")

for d, deg in [(2, 2), (4, 3), (4, 4), (8, 5)]:
    check(f"Grover  d={d} deg={deg} unitary", unitary(coin_grover(d, deg)))
    check(f"Fourier d={d} deg={deg} unitary", unitary(coin_fourier(d, deg)))
    w = RNG.normal(size=deg)
    check(f"Householder d={d} deg={deg} unitary", unitary(coin_householder(d, deg, w=w)))
    th = RNG.uniform(0, 2 * np.pi, deg * (deg - 1) // 2)
    check(f"Param real d={d} deg={deg} unitary",
          unitary(coin_parametric(d, deg, theta=th)))
    ph = RNG.uniform(0, 2 * np.pi, deg)
    check(f"Param cplx d={d} deg={deg} unitary",
          unitary(coin_parametric(d, deg, theta=th, phase=ph)))

check("Hadamard d=2 unitary", unitary(coin_hadamard(2)))
check("Householder is real", np.allclose(coin_householder(4, 4, w=RNG.normal(size=4)).imag, 0))
check("Param with phase=None is real",
      np.allclose(coin_parametric(4, 4, theta=RNG.uniform(0, 6, 6)).imag, 0))
check("Param with phases is genuinely complex",
      not np.allclose(coin_parametric(4, 4, theta=RNG.uniform(0, 6, 6),
                                      phase=RNG.uniform(0.5, 2.5, 4)).imag, 0))
check("Grover deg=2 equals the X gate (2|s><s|-I on 2 dims)",
      np.allclose(coin_grover(2, 2), np.array([[0, 1], [1, 0]])))
check("coin acts as identity outside the true degree",
      np.allclose(coin_grover(4, 2)[2:, 2:], np.eye(2)))


# ------------------------------------------------- shift and step ----------
print("\n[2] shift is an involutive permutation; step operators are unitary")

graphs = {
    "cycle-8": nx.cycle_graph(8),
    "path-7": nx.path_graph(7),
    "petersen": nx.petersen_graph(),
    "random-G(12,0.3)": nx.gnp_random_graph(12, 0.3, seed=7),
    "star-6": nx.star_graph(6),
}

for name, g in graphs.items():
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    w = DTQW(g)
    S = w.shift_matrix()
    check(f"{name}: S unitary", unitary(S))
    check(f"{name}: S^2 = I (flip-flop involution)",
          np.allclose(S @ S, np.eye(w.dim)))
    C = w.coin_matrix(fixed(coin_grover))
    check(f"{name}: C unitary", unitary(C))
    U = w.step_matrix(fixed(coin_grover))
    check(f"{name}: U = S(I(x)C) unitary", unitary(U))
    Uc = w.step_matrix(fixed(coin_fourier))
    check(f"{name}: complex-coin step unitary", unitary(Uc))


# ------------------------------------- independent reference walk ----------
print("\n[3] matrix evolution agrees with an independently written walk")


def reference_walk(graph, steps, start, coin_blocks):
    """Amplitude-by-amplitude flip-flop walk. Shares no code with dtqw.py."""
    g = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    n = g.number_of_nodes()
    nbrs = [sorted(g.neighbors(v)) for v in range(n)]
    pos = {v: {u: i for i, u in enumerate(nbrs[v])} for v in range(n)}
    d = coin_blocks[0].shape[0]

    amp = {}
    deg0 = len(nbrs[start])
    for i in range(deg0):
        amp[(start, i)] = 1.0 / np.sqrt(deg0)

    for _t in range(steps):
        # coin
        after_coin = {}
        for v in range(n):
            vec = np.zeros(d, dtype=complex)
            for c in range(d):
                vec[c] = amp.get((v, c), 0.0)
            if not np.any(vec):
                continue
            vec = coin_blocks[v] @ vec
            for c in range(d):
                if abs(vec[c]) > 1e-15:
                    after_coin[(v, c)] = vec[c]
        # flip-flop shift
        nxt = {}
        for (v, c), a in after_coin.items():
            if c < len(nbrs[v]):
                u = nbrs[v][c]
                j = pos[u][v]
                nxt[(u, j)] = nxt.get((u, j), 0.0) + a
            else:
                nxt[(v, c)] = nxt.get((v, c), 0.0) + a
        amp = nxt
    return amp


for name, g in [("cycle-8", nx.cycle_graph(8)),
                ("petersen", nx.petersen_graph()),
                ("random-G(12,0.3)", nx.gnp_random_graph(12, 0.3, seed=7))]:
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    w = DTQW(g)
    blocks = [coin_grover(w.coin_dim, int(w.degrees[v])) for v in range(w.n_nodes)]
    ref = reference_walk(g, 5, 0, blocks)
    psi = w.walk(fixed(coin_grover), steps=5, start=0)
    diff = 0.0
    for v in range(w.n_nodes):
        for c in range(w.coin_dim):
            diff = max(diff, abs(psi[w.index(v, c)] - ref.get((v, c), 0.0)))
    check(f"{name}: matches reference walk (max |delta| = {diff:.2e})", diff < 1e-10)


# ------------------------------------------------ norm and ballistics ------
print("\n[4] norm is conserved and spreading is ballistic, not diffusive")

N, T = 129, 40
cyc = nx.cycle_graph(N)
w = DTQW(cyc)
start = N // 2

psi = w.initial_state(start, coin_state=np.array([1, 1j]) / np.sqrt(2))
sigmas = []
for t in range(1, T + 1):
    psi = w.step_matrix(fixed(coin_hadamard)) @ psi
    p = w.position_distribution(psi)
    x = np.arange(N) - start
    mean = float(p @ x)
    sigmas.append(float(np.sqrt(p @ (x - mean) ** 2)))

check(f"norm conserved after {T} steps (|1 - ||psi|| | = {abs(1 - np.linalg.norm(psi)):.2e})",
      abs(1 - np.linalg.norm(psi)) < 1e-10)
check("walker stayed inside the cycle (no wrap contamination)",
      w.position_distribution(psi)[[0, 1, N - 2, N - 1]].sum() < 1e-9)

t_arr = np.arange(1, T + 1)
lin = np.polyfit(t_arr[10:], np.array(sigmas)[10:], 1)
log_slope = np.polyfit(np.log(t_arr[10:]), np.log(np.array(sigmas)[10:]), 1)[0]
print(f"        sigma(t) linear fit slope = {lin[0]:.4f};  log-log slope = {log_slope:.4f}")
check("spreading exponent ~ 1 (ballistic), not 0.5 (diffusive)",
      0.93 < log_slope < 1.05)
check("sigma/t in the Hadamard-walk range",
      0.45 < sigmas[-1] / T < 0.62)

# classical comparison on the same graph
P = np.zeros((N, N))
for v in range(N):
    for u in cyc.neighbors(v):
        P[u, v] = 1.0 / cyc.degree(v)
pc = np.zeros(N)
pc[start] = 1.0
for _t in range(T):
    pc = P @ pc
x = np.arange(N) - start
sig_c = float(np.sqrt(pc @ (x - pc @ x) ** 2))
print(f"        quantum sigma({T}) = {sigmas[-1]:.3f}   classical sigma({T}) = {sig_c:.3f}")
check("quantum spreads faster than classical", sigmas[-1] > 2.5 * sig_c)


# ------------------------------------------------------- circuits ----------
print("\n[5] the Qiskit circuit reproduces the matrix evolution")

from qiskit.quantum_info import Statevector

for name, g, steps in [("cycle-8", nx.cycle_graph(8), 4),
                       ("petersen", nx.petersen_graph(), 3)]:
    w = DTQW(g)
    qc = w.circuit(fixed(coin_grover), steps=steps, start=0)
    sv = np.asarray(Statevector(qc).data)
    psi = w.walk(fixed(coin_grover), steps=steps, start=0)
    check(f"{name}: circuit statevector == matrix evolution "
          f"(max |delta| = {np.abs(sv - psi).max():.2e})",
          np.allclose(sv, psi, atol=1e-8))
    print(f"        {name}: {w.n_qubits} qubits "
          f"({w.n_pos_qubits} position + {w.n_coin_qubits} coin), dim {w.dim}")


# ----------------------------------------------- the two axes bite ---------
print("\n[6] both modification axes actually change the walk")

w = DTQW(nx.cycle_graph(65))
s0 = w.walk(fixed(coin_grover), steps=20, start=32)
p0 = w.position_distribution(s0)

s_struct = w.walk(structure_dependent(lambda deg: np.pi / 4 + 0.1 * deg),
                  steps=20, start=32)
s_step = w.walk(step_dependent(lambda t: np.pi / 4 + 0.05 * t), steps=20, start=32)
s_cplx = w.walk(structure_dependent(lambda deg: np.pi / 4 + 0.1 * deg,
                                    complex_phases=True), steps=20, start=32)

tvd = lambda a, b: 0.5 * np.abs(a - b).sum()
p_struct = w.position_distribution(s_struct)
p_step = w.position_distribution(s_step)
p_cplx = w.position_distribution(s_cplx)

print(f"        TVD(Grover, structure-dependent) = {tvd(p0, p_struct):.4f}")
print(f"        TVD(Grover, step-dependent)      = {tvd(p0, p_step):.4f}")
print(f"        TVD(real, complex) same angles   = {tvd(p_struct, p_cplx):.4f}")
check("structure-dependent coin changes the distribution", tvd(p0, p_struct) > 1e-3)
check("step-dependent coin changes the distribution", tvd(p0, p_step) > 1e-3)
check("complex phases change the distribution (complex coin is not equivalent to a real one)",
      tvd(p_struct, p_cplx) > 1e-3)

from dtqw import coin_parametric as _cp


def _uniform_phase_coin(angle, ph):
    def fn(d, deg, vertex=None, step=None, **kw):
        n_ang = max(1, deg * (deg - 1) // 2)
        return _cp(d, deg, theta=np.full(n_ang, angle),
                   phase=np.full(deg, ph))
    return fn


p_g0 = w.position_distribution(w.walk(_uniform_phase_coin(0.7, 0.0), 20, 32))
p_g1 = w.position_distribution(w.walk(_uniform_phase_coin(0.7, 1.3), 20, 32))
check(f"gauge control: a uniform coin phase is unobservable "
      f"(TVD = {tvd(p_g0, p_g1):.2e})", tvd(p_g0, p_g1) < 1e-12)

# a control: on a degree-regular graph a *degree*-dependent coin must do nothing
w_reg = DTQW(nx.petersen_graph())
a = w_reg.position_distribution(w_reg.walk(structure_dependent(lambda deg: 0.7), 6, 0))
b = w_reg.position_distribution(w_reg.walk(structure_dependent(lambda deg: 0.7), 6, 0))
check("control: same coin, same result (determinism)", np.allclose(a, b))


# ------------------------------------------------------------ summary ------
n_pass = sum(1 for _n, c in results if c)
print(f"\n{'=' * 62}\n{n_pass}/{len(results)} checks passed\n{'=' * 62}")
if n_pass != len(results):
    print("FAILURES:")
    for n, c in results:
        if not c:
            print("  -", n)
    raise SystemExit(1)
