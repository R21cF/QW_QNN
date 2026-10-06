# Exported from qwqnn_lat_colab.ipynb by nb/export_cells.py; do not edit here.
# ---- qml-benchmarks models (the suite's own implementations) --------------------------
# Kernel models: the Gram matrix for each non-C hyperparameter setting is computed once on the training set
# (and once test x train) with the model's own precompute_kernel, then C is cross-validated on slices of it.
# For +-1 bit features the suite's MinMaxScaler is identical on every fold, so the slices equal what refitting
# would compute (for PQK, gamma's data-dependent default is taken from the full training set).
import itertools, warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC as SkSVC
from sklearn.ensemble import RandomForestClassifier
try:
    import qml_benchmarks.models as QB          # needed only for the suite's own models (iqp_kernel, projected_kernel, data_reuploading, mlp, svc_rbf)
except ImportError:                              # the walk QNN, LAT kernel and their *_full selectors need only numpy/scipy/sklearn
    QB = None

CV = StratifiedKFold(5, shuffle=True, random_state=42)
C_GRID = [0.1, 1, 10, 100]                                    # suite grid for C


def _kernel_arm(make_model, settings, Xtr, ytr, Xte):
    best = (-1, None, None)
    for st in settings:
        m = make_model(**st)
        m.initialize(Xtr.shape[1], np.array([-1, 1]))
        m.scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2)).fit(Xtr)
        A, B = m.transform(Xtr), m.transform(Xte)
        Ktr = np.asarray(m.precompute_kernel(A, A)); Kte = np.asarray(m.precompute_kernel(B, A))
        for C in C_GRID:
            acc = np.mean([(SkSVC(kernel="precomputed", C=C).fit(Ktr[np.ix_(i, i)], ytr[i])
                            .predict(Ktr[np.ix_(j, i)]) == ytr[j]).mean() for i, j in CV.split(Xtr, ytr)])
            if acc > best[0]:
                best = (acc, dict(st, C=C), (Ktr, Kte))
    acc, hp, (Ktr, Kte) = best
    pred = SkSVC(kernel="precomputed", C=hp["C"]).fit(Ktr, ytr).predict(Kte)
    return acc, hp, pred


def iqp_kernel(Xtr, ytr, Xte):
    return _kernel_arm(lambda **k: QB.IQPKernelClassifier(jit=True, **k),
                       [{"repeats": r} for r in [1, 5, 10]], Xtr, ytr, Xte)


def projected_kernel(Xtr, ytr, Xte):
    sets = [{"gamma_factor": g, "t": t, "trotter_steps": s}
            for g, t, s in itertools.product([0.1, 1, 10], [0.01, 0.1, 1.0], [1, 3, 5])]
    return _kernel_arm(lambda **k: QB.ProjectedQuantumKernel(**k), sets, Xtr, ytr, Xte)


def _gridsearch(est, grid, Xtr, ytr, Xte):
    gs = GridSearchCV(est, grid, cv=CV, n_jobs=1).fit(Xtr, ytr)
    return gs.best_score_, {k: (v if np.isscalar(v) else str(v)) for k, v in gs.best_params_.items()}, gs.predict(Xte)


def data_reuploading(Xtr, ytr, Xte):
    # reduced from the suite grid (3 lr x 4 layers x 3 observables) to keep the run tractable
    grid = {"learning_rate": [0.01, 0.1], "n_layers": [1, 5], "observable_type": ["half", "full"]}
    return _gridsearch(QB.DataReuploadingClassifier(random_state=42), grid, Xtr, ytr, Xte)


def mlp(Xtr, ytr, Xte):
    grid = {"learning_rate_init": [0.001, 0.01, 0.1],
            "hidden_layer_sizes": [(100,), (10, 10, 10, 10), (50, 10, 5)], "alpha": [0.01, 0.001, 0.0001]}
    return _gridsearch(QB.MLPClassifier(random_state=42), grid, Xtr, ytr, Xte)


def svc_rbf(Xtr, ytr, Xte):
    return _gridsearch(QB.SVC(), {"gamma": [0.001, 0.01, 0.1, 1], "C": C_GRID}, Xtr, ytr, Xte)


def random_forest(Xtr, ytr, Xte):
    grid = {"n_estimators": [100, 300], "max_depth": [None, 4, 8]}
    return _gridsearch(RandomForestClassifier(random_state=42), grid, Xtr, ytr, Xte)


def walk_qnn(atr, ytr, ate, p, walk=True):
    grid = {"h": [3, 4, 5, 6], "t": [1, 2, 3, 4, 6, 8, 12, 16] if walk else [0],
            "ridge": [1e-3, 1e-2, 1e-1, 1.0]}
    return _gridsearch(HadamardWalkQNN(p=p), grid, atr.reshape(-1, 1), ytr, ate.reshape(-1, 1))


def lat_kernel(atr, ytr, ate, p, n):
    grid = {"k": [n - 2, n - 3, n - 4, n - 5, n - 6], "C": C_GRID}
    return _gridsearch(LATKernelSVM(p=p), grid, atr.reshape(-1, 1), ytr, ate.reshape(-1, 1))


def ct_walk_qnn(atr, ytr, ate, p):
    grid = {"h": [3, 4, 5, 6], "t": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
            "ridge": [1e-3, 1e-2, 1e-1, 1.0]}
    return _gridsearch(CTWalkQNN(p=p), grid, atr.reshape(-1, 1), ytr, ate.reshape(-1, 1))


# ---- uncapped setup (PREREGISTRATION_2): walk resolution h in {3..n}; LAT kernel over every width -----------------
# CV uses the exact dual (kernel-ridge) form of CTWalkQNN.fit (identical selection to GridSearchCV; checked on old draws)
FULL_T = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
FULL_RIDGE = [1e-3, 1e-2, 1e-1, 1.0]


def _dual_fit_predict(K_tr, K_te, y, lam):
    m_tr = K_tr.mean(1); mm = K_tr.mean()
    Kc = K_tr - m_tr[:, None] - m_tr[None, :] + mm
    alpha = np.linalg.solve(Kc + lam * np.eye(len(y)), y - y.mean())
    Kt = K_te - K_te.mean(1)[:, None] - m_tr[None, :] + mm
    return Kt @ alpha + y.mean()


def ct_walk_qnn_full(atr, ytr, ate, p, n, t_grid=FULL_T):
    y = ytr.astype(float)
    splits = list(CV.split(atr.reshape(-1, 1), y))
    best = (-1, None)
    for h in range(3, n + 1):
        for t in t_grid:
            P = CTWalkQNN(p=p, h=h, t=t)._probs(atr.reshape(-1, 1)); K = P @ P.T
            for lam in FULL_RIDGE:
                acc = np.mean([(np.where(_dual_fit_predict(K[np.ix_(i, i)], K[np.ix_(j, i)], y[i], lam) >= 0, 1, -1)
                                == y[j]).mean() for i, j in splits])
                key = (h, lam, t)
                if acc > best[0] + 1e-12 or (abs(acc - best[0]) <= 1e-12 and key < best[1]):
                    best = (acc, key)
    acc, (h, lam, t) = best
    pred = CTWalkQNN(p=p, h=h, t=t, ridge=lam).fit(atr.reshape(-1, 1), y).predict(ate.reshape(-1, 1))
    return acc, {"h": h, "t": t, "ridge": lam}, pred


def lat_kernel_full(atr, ytr, ate, p, n):
    grid = {"k": list(range(1, n)), "C": [0.01, 0.1, 1, 10, 100, 1000]}
    return _gridsearch(LATKernelSVM(p=p), grid, atr.reshape(-1, 1), ytr, ate.reshape(-1, 1))
