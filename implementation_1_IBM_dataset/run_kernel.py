"""
Quantum kernel comparison: the walk map against the standard product and
entangling maps, and against classical kernels.

Protocol, applied identically to every quantum map so that no map is
advantaged by its tuning:
  * features standardised, then reduced by PCA to a power of two when needed
  * bandwidth lambda selected from a fixed grid by 3-fold cross-validation on
    the TRAINING fold only, jointly with the SVM's C
  * the selected model refitted on the full training fold and scored once on
    the held-out test fold
  * five stratified splits, so every number carries a spread

The kernel is the fidelity kernel K(x,y) = |<Phi(x)|Phi(y)>|^2, computed
exactly from statevectors. Positive semi-definiteness is checked rather than
assumed: a fidelity kernel is PSD by construction, and a violation beyond
numerical tolerance would mean a bug.
"""

from __future__ import annotations

import json
import time

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import featuremaps as FM

LAMBDAS = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]
CS = [0.1, 1.0, 10.0, 100.0]
# depth of the encoding: repetitions for the Z/ZZ maps, layers for the walk.
# Tuned on the same grid for every quantum map, so no map is handicapped by
# a repetition count chosen for a different one.
DEPTHS = [1, 2, 3]
N_SPLITS = 5
RNG = np.random.default_rng(0)


# ------------------------------------------------------------- datasets --


def _prep(X, y, n_feat, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    if len(y) > n_samples:
        idx = rng.permutation(len(y))[:n_samples]
        X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    if X.shape[1] != n_feat:
        X = PCA(n_components=n_feat, random_state=0).fit_transform(X)
        X = StandardScaler().fit_transform(X)
    return X, y


def datasets():
    out = {}

    from qiskit_machine_learning.datasets import ad_hoc_data
    Xa, ya, Xb, yb = ad_hoc_data(training_size=60, test_size=30, n=2,
                                 gap=0.3, one_hot=False)
    X = np.vstack([Xa, Xb]); y = np.concatenate([ya, yb]).astype(int)
    out["ad hoc (2f)"] = (StandardScaler().fit_transform(X), y, 2)

    d = load_iris()
    m = d.target != 0                      # versicolor vs virginica: the hard pair
    out["iris v-v (4f)"] = (*_prep(d.data[m], d.target[m], 4, 200), 4)

    d = load_wine()
    m = d.target != 2
    out["wine 0-1 (4f)"] = (*_prep(d.data[m], d.target[m], 4, 200), 4)

    d = load_breast_cancer()
    out["breast cancer (8f)"] = (*_prep(d.data, d.target, 8, 200), 8)

    d = load_digits()
    m = np.isin(d.target, [3, 8])
    out["digits 3v8 (8f)"] = (*_prep(d.data[m], (d.target[m] == 8).astype(int),
                                     8, 200), 8)
    return out


# --------------------------------------------------------------- kernel --


def fidelity_kernel(S1, S2):
    return np.abs(S1 @ S2.conj().T) ** 2


def quantum_eval(kind, X, y, n_feat, seed):
    """Tune (lambda, C) by CV on the training fold, score once on test."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(sss.split(X, y))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    best = (-1, None, None, None)
    for dep in DEPTHS:
        for lam in LAMBDAS:
            fm = FM.make(kind, n_feat, scale=lam, reps=dep, layers=dep)
            Str = fm.states(Xtr)
            Ktr = fidelity_kernel(Str, Str)
            for C in CS:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                sc = []
                for a, b in skf.split(Xtr, ytr):
                    clf = SVC(kernel="precomputed", C=C)
                    clf.fit(Ktr[np.ix_(a, a)], ytr[a])
                    sc.append(clf.score(Ktr[np.ix_(b, a)], ytr[b]))
                m = float(np.mean(sc))
                if m > best[0]:
                    best = (m, lam, C, dep)

    _cv, lam, C, dep = best
    fm = FM.make(kind, n_feat, scale=lam, reps=dep, layers=dep)
    Str, Ste = fm.states(Xtr), fm.states(Xte)
    Ktr = fidelity_kernel(Str, Str)
    clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
    acc = clf.score(fidelity_kernel(Ste, Str), yte)

    ev = np.linalg.eigvalsh((Ktr + Ktr.T) / 2)
    return dict(acc=float(acc), lam=lam, C=C, depth=dep, cv=best[0],
                min_eig=float(ev.min()), qubits=fm.n_qubits,
                offdiag_mean=float(Ktr[~np.eye(len(Ktr), dtype=bool)].mean()))


def classical_eval(kind, X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(sss.split(X, y))
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    grid = ([dict(C=c, gamma=g) for c in CS
             for g in ["scale", 0.01, 0.1, 1.0]] if kind == "rbf"
            else [dict(C=c) for c in CS])
    best = (-1, None)
    for p in grid:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        sc = []
        for a, b in skf.split(Xtr, ytr):
            clf = SVC(kernel=kind, **p).fit(Xtr[a], ytr[a])
            sc.append(clf.score(Xtr[b], ytr[b]))
        m = float(np.mean(sc))
        if m > best[0]:
            best = (m, p)
    clf = SVC(kernel=kind, **best[1]).fit(Xtr, ytr)
    return dict(acc=float(clf.score(Xte, yte)), params=best[1], cv=best[0])


# ------------------------------------------------------------------ main --

if __name__ == "__main__":
    METHODS = [("walk", "quantum"), ("z", "quantum"), ("zz", "quantum"),
               ("rbf", "classical"), ("linear", "classical")]
    LABEL = {"walk": "walk kernel", "z": "Z kernel", "zz": "ZZ kernel",
             "rbf": "classical RBF", "linear": "classical linear"}

    rows = []
    for dname, (X, y, nf) in datasets().items():
        print(f"\n=== {dname}   n={len(y)}  features={X.shape[1]} ===")
        for kind, fam in METHODS:
            t0 = time.time()
            accs, extra = [], []
            for s in range(N_SPLITS):
                r = (quantum_eval(kind, X, y, nf, s) if fam == "quantum"
                     else classical_eval(kind, X, y, s))
                accs.append(r["acc"]); extra.append(r)
            mu, sd = float(np.mean(accs)) * 100, float(np.std(accs)) * 100
            qb = extra[0].get("qubits", "--")
            me = min(e.get("min_eig", 0.0) for e in extra)
            od = float(np.mean([e.get("offdiag_mean", np.nan) for e in extra]))
            rows.append(dict(dataset=dname, method=LABEL[kind], family=fam,
                             acc_mean=mu, acc_std=sd, qubits=qb,
                             min_eig=me, offdiag=od,
                             lams=[e.get("lam") for e in extra],
                             depths=[e.get("depth") for e in extra]))
            tail = (f"  qubits={qb}  min eig={me:+.2e}  "
                    f"mean off-diag K={od:.3f}" if fam == "quantum" else "")
            print(f"  {LABEL[kind]:<18} {mu:5.1f} +/- {sd:4.1f}{tail}"
                  f"   [{time.time() - t0:.0f}s]")

    json.dump(rows, open("kernel_results.json", "w"), indent=2)
    print("\nwrote kernel_results.json")
