"""Benchmark the walk kernel against tuned classical baselines."""
from __future__ import annotations
import itertools, json, sys, time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, "qmlb")
import walkmap as W
import qkernels as Q

N_SPLITS, N_INNER, SEED = 10, 3, 0
CS = [0.1, 1.0, 10.0, 100.0]


# ----------------------------------------------------------------- data --
def datasets(seed=SEED):
    np.random.seed(seed)
    from two_curves import generate_two_curves
    from linearly_separable import generate_linearly_separable
    from hidden_manifold import generate_hidden_manifold_model
    from hyperplanes import generate_hyperplanes_parity
    D = {}
    X, y = generate_two_curves(300, 4, 5, 0.1, 0.01);         D["two_curves(4f)"] = (np.array(X), np.array(y))
    X, y = generate_linearly_separable(300, 4, 0.02);         D["lin_sep(4f)"] = (np.array(X), np.array(y))
    X, y = generate_hidden_manifold_model(300, 4, 2);         D["hidden_manifold(4f)"] = (np.array(X), np.array(y))
    X, y = generate_hyperplanes_parity(300, 4, 3, 4);         D["hyperplanes_parity(4f)"] = (np.array(X), np.array(y))
    return {k: (np.asarray(a, float), np.asarray(b).ravel()) for k, (a, b) in D.items()}


# ---------------------------------------------------------------- kernels --
def periodic_gram(A, B, gamma, period):
    """ExpSineSquared: exp(-gamma * sum_j sin^2(pi |a_j - b_j| / period))."""
    d = A[:, None, :] - B[None, :, :]
    return np.exp(-gamma * (np.sin(np.pi * np.abs(d) / period) ** 2).sum(-1))


def rbf_gram(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


WALK_GRID = [dict(lam=l, T=t, n=n, entry=e)
             for l in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
             for t in (2, 4, 8, 16, 32)
             for n in (3, 4, 5)
             for e in ("coin", "position")]
LAMS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# the existing quantum kernels, on the same encoding-scale grid as the walk map
Z_GRID  = [dict(lam=l, reps=r) for l in LAMS for r in (1, 2, 3, 4)]
ZZ_GRID = [dict(lam=l, reps=r) for l in LAMS for r in (1, 2, 3, 4)]
RBF_GRID = [dict(gamma=g) for g in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
PER_GRID = [dict(gamma=g, period=p) for g in (0.1, 0.3, 1.0, 3.0)
            for p in (0.5, 1.0, 2.0, 4.0)]


def gram_fn(method, p):
    if method == "walk":
        return lambda A, B: W.walk_kernel(A, B, **p)
    if method == "z":
        return lambda A, B: Q.z_kernel(A, B, **p)
    if method == "zz":
        return lambda A, B: Q.zz_kernel(A, B, **p)
    if method == "rbf":
        return lambda A, B: rbf_gram(A, B, p["gamma"])
    if method == "periodic":
        return lambda A, B: periodic_gram(A, B, p["gamma"], p["period"])
    if method == "linear":
        return lambda A, B: A @ B.T
    raise ValueError(method)


GRIDS = {"walk": WALK_GRID, "z": Z_GRID, "zz": ZZ_GRID,
         "rbf": RBF_GRID, "periodic": PER_GRID, "linear": [dict()]}
ORDER = ["walk", "z", "zz", "rbf", "periodic", "linear"]
LABEL = {"walk": "walk (ours)", "z": "Z map", "zz": "ZZ map",
         "rbf": "classical RBF", "periodic": "classical periodic", "linear": "classical linear"}


def run_method(method, Xtr, ytr, Xte, yte, rng):
    skf = StratifiedKFold(N_INNER, shuffle=True, random_state=rng)
    best = (-1.0, None, None)
    for p in GRIDS[method]:
        g = gram_fn(method, p)
        K = g(Xtr, Xtr)
        for C in CS:
            sc = []
            for a, b in skf.split(Xtr, ytr):
                clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(a, a)], ytr[a])
                sc.append(clf.score(K[np.ix_(b, a)], ytr[b]))
            m = float(np.mean(sc))
            if m > best[0]:
                best = (m, p, C)
    _, p, C = best
    g = gram_fn(method, p)
    Ktr = g(Xtr, Xtr)
    clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
    acc = float(clf.score(g(Xte, Xtr), yte))
    off = float(Ktr[~np.eye(len(ytr), dtype=bool)].mean())
    return acc, dict(best, **{}) if False else dict(params=p, C=C, cv=best[0], offdiag=off)


def main():
    out = {}
    for name, (X, y) in datasets().items():
        X = StandardScaler().fit_transform(X)
        sss = StratifiedShuffleSplit(N_SPLITS, test_size=0.3, random_state=SEED)
        rows = {m: [] for m in GRIDS}
        meta = {m: [] for m in GRIDS}
        for i, (tr, te) in enumerate(sss.split(X, y)):
            for m in GRIDS:
                t0 = time.time()
                acc, info = run_method(m, X[tr], y[tr], X[te], y[te], SEED + i)
                rows[m].append(acc); meta[m].append(info)
        print(f"\n=== {name}   N={len(y)} d={X.shape[1]} majority={max(np.bincount((y>0).astype(int)))/len(y)*100:.1f}% ===")
        from collections import Counter
        wa = np.array(rows["walk"]) * 100
        for m in ORDER:
            a = np.array(rows[m]) * 100
            extra = ""
            if m != "walk":
                from scipy import stats
                d = wa - a
                pv = stats.ttest_rel(wa, a).pvalue if d.std() > 0 else 1.0
                extra = f"   walk-{m} {d.mean():+5.1f} pp  p={pv:.3f}"
            pk = Counter(str(x["params"]) for x in meta[m]).most_common(1)[0]
            print(f"  {LABEL[m]:19s} {a.mean():5.1f} +/- {a.std():4.1f}{extra}")
            if m == "walk":
                ec = Counter(x["params"]["entry"] for x in meta[m])
                print(f"    modal config {pk[0]} ({pk[1]}/{N_SPLITS});"
                      f" entry picks {dict(ec)}")
        out[name] = {m: dict(acc=list(np.array(rows[m])*100), meta=meta[m]) for m in GRIDS}
    json.dump(out, open("bench_results.json", "w"), indent=1)
    print("\nwrote bench_results.json")


if __name__ == "__main__":
    main()
