"""Is the optimum at the grid boundary because the map degenerates, or because the
grid was too narrow?  Extend it in both directions and look at the whole surface."""
import sys, numpy as np
sys.path.insert(0, "qmlb")
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import walkmap as W
from bench import datasets, CS

LAMS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
TS   = [2, 4, 8, 16, 32]
NS   = [2, 3, 4, 5, 6]

def cv_acc(K, y, rng=0):
    skf = StratifiedKFold(3, shuffle=True, random_state=rng)
    best = -1
    for C in CS:
        sc = [SVC(kernel="precomputed", C=C).fit(K[np.ix_(a,a)], y[a]).score(K[np.ix_(b,a)], y[b])
              for a, b in skf.split(K, y)]
        best = max(best, float(np.mean(sc)))
    return best

for name in ("two_curves(4f)", "hyperplanes_parity(4f)"):
    X, y = datasets()[name]
    X = StandardScaler().fit_transform(X)
    tr, te = next(iter(StratifiedShuffleSplit(1, test_size=0.3, random_state=0).split(X, y)))
    Xtr, ytr = X[tr], y[tr]
    print(f"\n########## {name} — inner-CV accuracy (%), entry=coin ##########")
    print("      T=2   T=4   T=8  T=16  T=32      (n = position qubits)")
    for n in NS:
        for lam in LAMS:
            row = []
            for T in TS:
                K = W.walk_kernel(Xtr, lam=lam, T=T, n=n, entry="coin")
                row.append(cv_acc(K, ytr) * 100)
            if lam in (0.02, 0.1, 0.5, 2.0, 8.0):
                print(f"n={n} lam={lam:<5g} " + "  ".join(f"{v:4.1f}" for v in row))
        print()
