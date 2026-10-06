"""Model selection for the discrete-logarithm task (thesis Sections 4.4, 5.1.1 and 4.2.3).

Every selector takes training inputs, training labels and test inputs, chooses hyperparameters by five-fold
stratified cross-validation on the training set alone (fold assignment seeded at 42), refits on the whole training
set and returns (cv_accuracy, hyperparameters, test_predictions); the test set is scored once, by the caller.

Classical comparators: the multilayer perceptron and RBF support vector machine of the benchmark suite of Bowles et
al. (qml-benchmarks), with the suite's grids. The suite's `MLPClassifier` and `SVC` are scikit-learn's with
`max_iter=3000` and `kernel="rbf"` respectively, so they are built here from scikit-learn directly and the suite
(PennyLane, JAX) is not needed. The random forest is not part of the suite; its grid is this thesis's.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .models import CTWalkQNN, HadamardWalkQNN, LATKernelSVM, dual_fit_predict

CV = StratifiedKFold(5, shuffle=True, random_state=42)

# walk classifier (Section 4.4): h in {3..n}, t, ridge penalty
T_GRID = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0]
# Liu et al.'s kernel: every width k in {1..n-1}, margin C
LAT_C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
# coined Hadamard-walk control, capped at h <= 6, integer step counts
HADAMARD_H_GRID = [3, 4, 5, 6]
HADAMARD_T_GRID = [1, 2, 3, 4, 6, 8, 12, 16]
# first confirmatory run (PREREGISTRATION.md): the walk capped at h <= 6
CAPPED_H_GRID = [3, 4, 5, 6]
# qml-benchmarks grids
SUITE_C_GRID = [0.1, 1, 10, 100]
MLP_GRID = {"learning_rate_init": [0.001, 0.01, 0.1],
            "hidden_layer_sizes": [(100,), (10, 10, 10, 10), (50, 10, 5)], "alpha": [0.01, 0.001, 0.0001]}
SVC_GRID = {"gamma": [0.001, 0.01, 0.1, 1], "C": SUITE_C_GRID}
RF_GRID = {"n_estimators": [100, 300], "max_depth": [None, 4, 8]}


def _clean(hp):
    return {k: (v.item() if hasattr(v, "item") else v if np.isscalar(v) or v is None else str(v)) for k, v in hp.items()}


def gridsearch(est, grid, Xtr, ytr, Xte, n_jobs=1):
    gs = GridSearchCV(est, grid, cv=CV, n_jobs=n_jobs).fit(Xtr, ytr)
    return gs.best_score_, _clean(gs.best_params_), gs.predict(Xte)


# ---- classical comparators, on +-1 bit features ----------------------------------------------------------
def mlp(Xtr, ytr, Xte, n_jobs=1):
    return gridsearch(MLPClassifier(max_iter=3000, random_state=42), MLP_GRID, Xtr, ytr, Xte, n_jobs)


def svc_rbf(Xtr, ytr, Xte, n_jobs=1):
    return gridsearch(SVC(kernel="rbf"), SVC_GRID, Xtr, ytr, Xte, n_jobs)


def random_forest(Xtr, ytr, Xte, n_jobs=1):
    return gridsearch(RandomForestClassifier(random_state=42), RF_GRID, Xtr, ytr, Xte, n_jobs)


CLASSICAL = {"MLP": mlp, "SVM (RBF)": svc_rbf, "Random forest": random_forest}


# ---- quantum pipelines, on the exponents a = log_g x ------------------------------------------------------
def ct_walk_qnn(atr, ytr, ate, p, n, t_grid=T_GRID, h_grid=None):
    """The walk classifier, h in {3..n}. Cross-validation uses the exact dual form of the fit (the walk kernel);
    candidates are visited in GridSearchCV's order (h, ridge, t) and the first best is kept, as GridSearchCV does.
    The final model is the primal CTWalkQNN refitted on the whole training set."""
    y = np.asarray(ytr, float)
    X = np.asarray(atr).reshape(-1, 1)
    splits = list(CV.split(X, y))
    best = (-1, None)
    for h in (h_grid or range(3, n + 1)):
        for t in t_grid:
            P = CTWalkQNN(p=p, h=h, t=t)._probs(X); K = P @ P.T
            for lam in RIDGE_GRID:
                acc = np.mean([(np.where(dual_fit_predict(K[np.ix_(i, i)], K[np.ix_(j, i)], y[i], lam) >= 0, 1, -1)
                                == y[j]).mean() for i, j in splits])
                key = (h, lam, t)
                if acc > best[0] + 1e-12 or (abs(acc - best[0]) <= 1e-12 and key < best[1]):
                    best = (acc, key)
    acc, (h, lam, t) = best
    pred = CTWalkQNN(p=p, h=h, t=t, ridge=lam).fit(X, y).predict(np.asarray(ate).reshape(-1, 1))
    return acc, {"h": h, "t": t, "ridge": lam}, pred


def no_walk(atr, ytr, ate, p, n):
    """Ablation: the same classifier with the walk removed (t = 0), same h and ridge grids."""
    return ct_walk_qnn(atr, ytr, ate, p, n, t_grid=[0.0])


def lat_kernel(atr, ytr, ate, p, n):
    """Liu et al.'s kernel in a support vector machine, every width k in {1..n-1}."""
    return gridsearch(LATKernelSVM(p=p), {"k": list(range(1, n)), "C": LAT_C_GRID},
                      np.asarray(atr).reshape(-1, 1), ytr, np.asarray(ate).reshape(-1, 1))


def hadamard_walk_qnn(atr, ytr, ate, p, n=None):
    """Secondary control: the coined Hadamard walk in place of the continuous-time walk, h <= 6."""
    return gridsearch(HadamardWalkQNN(p=p), {"h": HADAMARD_H_GRID, "t": HADAMARD_T_GRID, "ridge": RIDGE_GRID},
                      np.asarray(atr).reshape(-1, 1), ytr, np.asarray(ate).reshape(-1, 1))


def ct_walk_qnn_capped(atr, ytr, ate, p, n=None):
    """First confirmatory run: the walk classifier restricted to h <= 6 (GridSearchCV, primal fits)."""
    return gridsearch(CTWalkQNN(p=p), {"h": CAPPED_H_GRID, "t": T_GRID, "ridge": RIDGE_GRID},
                      np.asarray(atr).reshape(-1, 1), ytr, np.asarray(ate).reshape(-1, 1))


def no_walk_capped(atr, ytr, ate, p, n=None):
    """First confirmatory run: the no-walk ablation restricted to h <= 6."""
    return gridsearch(CTWalkQNN(p=p, t=0.0), {"h": CAPPED_H_GRID, "ridge": RIDGE_GRID},
                      np.asarray(atr).reshape(-1, 1), ytr, np.asarray(ate).reshape(-1, 1))


def shots_accuracy(model_cls, hp, d, shots):
    """Test accuracy of the selected walk classifier when <O_w> is estimated from `shots` measurements per input."""
    m = model_cls(p=d["p"], shots=shots, **hp).fit(d["a_tr"].reshape(-1, 1), d["y_tr"])
    return float(np.mean(m.predict(d["a_te"].reshape(-1, 1)) == d["y_te"]))
