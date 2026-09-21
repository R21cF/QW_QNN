"""
Is the complex coin actually buying anything?

Axis 2 of the thesis claims that letting coin entries be complex reaches walk
behaviour a real coin cannot.  That is not self-evident: a uniform phase on a
coin block is a global phase and does nothing, so some of U(d) is gauge.
Before any application result rests on axis 2, the axis has to be shown
non-vacuous and its size measured rather than asserted.

The experiment is a reachability test.  Draw a target position distribution
from a Haar-random COMPLEX coin, then optimise over REAL orthogonal coins to
reproduce it.  The residual total variation distance is the part of
complex-coin behaviour the real family cannot reach.

Two controls decide whether a non-zero residual means anything:
  * real target    -- fitting a real target with real coins must reach ~0, or
                      the optimiser is the story rather than the coin family.
  * complex target, complex fit -- fitting a complex target with complex coins
                      must also reach ~0, confirming the target is reachable at
                      all and the parameterisation covers U(d).
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from scipy.stats import unitary_group

from dtqw import coin_parametric
from fastwalk import FastUniformWalk

RNG = np.random.default_rng(11)
STEPS = 8
N_TARGETS = 5
N_RESTARTS = 30


def real_coin(p, d):
    """Full O(d): Givens sweep (SO(d)) times a diagonal of signs."""
    n_ang = d * (d - 1) // 2
    m = coin_parametric(d, d, theta=p[:n_ang]).real
    s = np.where(p[n_ang:n_ang + d] >= 0, 1.0, -1.0)
    return m @ np.diag(s)


def complex_coin(p, d):
    n_ang = d * (d - 1) // 2
    return coin_parametric(d, d, theta=p[:n_ang], phase=p[n_ang:n_ang + d])


def tvd(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def fit(walk, target, start, builder, d, n_par, n_restarts=N_RESTARTS):
    """Minimise a smooth surrogate (squared Hellinger), report TVD at the
    optimum.  TVD itself is piecewise linear and defeats gradient-free
    simplex methods near the optimum."""
    def surrogate(p):
        q = walk.distribution(builder(p, d), STEPS, start)
        return float(((np.sqrt(np.maximum(q, 0)) - np.sqrt(target)) ** 2).sum())

    best_p, best = None, np.inf
    for _ in range(n_restarts):
        p0 = RNG.uniform(-np.pi, np.pi, n_par)
        r = minimize(surrogate, p0, method="Powell",
                     options={"maxfev": 40000, "ftol": 1e-13, "xtol": 1e-11})
        # polish: Powell stalls on the narrow valleys this surrogate has
        r2 = minimize(surrogate, r.x, method="L-BFGS-B",
                      options={"maxiter": 3000, "ftol": 1e-15, "gtol": 1e-12})
        p, f = (r2.x, float(r2.fun)) if r2.fun < r.fun else (r.x, float(r.fun))
        if f < best:
            best, best_p = f, p
    return tvd(walk.distribution(builder(best_p, d), STEPS, start), target)


# Every graph here is degree-regular with degree equal to the padded coin
# dimension.  Petersen (degree 3, coin dimension 4) was dropped: a coin acting
# on the padding direction leaks amplitude into a state the shift never moves,
# which is a different walk and confounds the comparison.
GRAPHS = [
    ("cycle-33 (deg 2)", nx.cycle_graph(33), 16),
    ("hypercube Q4 (deg 4)", nx.hypercube_graph(4), 0),
    ("4-regular-16 (deg 4)", nx.random_regular_graph(4, 16, seed=3), 0),
    ("4-regular-24 (deg 4)", nx.random_regular_graph(4, 24, seed=11), 0),
]

print(f"Can real coins reproduce complex-coin walks?  {STEPS} steps, "
      f"{N_TARGETS} targets, {N_RESTARTS} restarts")
print("=" * 86)
print(f"{'graph':<22} {'d':>2} {'dim O(d)':>9} {'dim U(d)':>9} "
      f"{'ctrl real':>10} {'ctrl cplx':>10} {'REAL FIT':>10}")
print("-" * 86)

rows = []
for name, g, start in GRAPHS:
    g = nx.convert_node_labels_to_integers(g, ordering="sorted")
    w = FastUniformWalk(g)
    d = w.d
    n_ang = d * (d - 1) // 2
    n_real, n_cplx = n_ang + d, n_ang + d

    # control 1: real target, real fit
    tgt_r = w.distribution(real_coin(RNG.uniform(-np.pi, np.pi, n_real), d),
                           STEPS, start)
    ctrl_real = fit(w, tgt_r, start, real_coin, d, n_real)

    # control 2 and the question, on the same Haar targets
    ctrl_cplx, real_fits = [], []
    for _t in range(N_TARGETS):
        blk = unitary_group.rvs(d, random_state=int(RNG.integers(1 << 30)))
        tgt = w.distribution(blk, STEPS, start)
        ctrl_cplx.append(fit(w, tgt, start, complex_coin, d, n_cplx))
        real_fits.append(fit(w, tgt, start, real_coin, d, n_real))

    rows.append((name, d, real_fits))
    print(f"{name:<22} {d:>2} {d * (d - 1) // 2:>9} {d * d:>9} "
          f"{ctrl_real:>10.2e} {float(np.median(ctrl_cplx)):>10.2e} "
          f"{float(np.median(real_fits)):>10.4f}")

print("-" * 86)
print("""
ctrl real  median residual TVD fitting a real target with real coins
ctrl cplx  median residual TVD fitting each complex target with complex coins
REAL FIT   median residual TVD fitting those same complex targets with REAL
           coins -- the behaviour axis 2 adds, if the two controls are ~0

dim O(d) counts Givens angles only; the fit also carries d sign parameters.
""")
print("per-graph spread of the real-coin residuals (sorted):")
for name, d, fits in rows:
    print(f"  {name:<22} " + "  ".join(f"{x:.4f}" for x in sorted(fits)))
