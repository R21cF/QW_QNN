"""Confirmatory run 2 (PREREGISTRATION_2.md). CTQW-QNN with h in {3..n}; LAT kernel as in run 1.
For speed, the CTQW model's CV is computed with the exact dual (kernel-ridge) form of the same fit, with an
unpenalised intercept; the final model is refitted with CTWalkQNN and its predictions are checked against the dual form."""
import json, os, time, warnings, itertools
import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
warnings.filterwarnings("ignore")
NB = os.environ.get("NB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nb"))   # exported from the notebook, see nb/README.md
exec(open(os.path.join(NB, "c2_data.py")).read())
exec(open(os.path.join(NB, "c3_models.py")).read())

CV = StratifiedKFold(5, shuffle=True, random_state=42)
SIZES, DRAWS = [8, 10, 12, 16], list(range(200, 210))
T_GRID = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
RIDGE = [1e-3, 1e-2, 1e-1, 1.0]
OUT = "lat_confirm2.json"


def dual_fit_predict(K_tr, K_te, y, lam):
    """Ridge with unpenalised intercept in dual form: centre the features implicitly through the Gram matrix."""
    m_tr = K_tr.mean(1); mm = K_tr.mean()
    Kc = K_tr - m_tr[:, None] - m_tr[None, :] + mm
    alpha = np.linalg.solve(Kc + lam * np.eye(len(y)), y - y.mean())
    m_te = K_te.mean(1)
    Kt = K_te - m_te[:, None] - m_tr[None, :] + mm
    return Kt @ alpha + y.mean()


def ctqw_select(d, n, t_grid):
    atr, ytr = d["a_tr"], d["y_tr"].astype(float)
    splits = list(CV.split(atr.reshape(-1, 1), ytr))
    best = (-1, None)
    for h in range(3, n + 1):                      # ParameterGrid order: h, ridge, t
        for t in t_grid:
            P = CTWalkQNN(p=d["p"], h=h, t=t)._probs(atr.reshape(-1, 1))
            K = P @ P.T
            for lam in RIDGE:
                acc = np.mean([(np.where(dual_fit_predict(K[np.ix_(i, i)], K[np.ix_(j, i)], ytr[i], lam) >= 0, 1, -1)
                                == ytr[j]).mean() for i, j in splits])
                key = (h, lam, t)
                if acc > best[0] + 1e-12 or (abs(acc - best[0]) <= 1e-12 and key < best[1]):
                    best = (acc, key)
    acc, (h, lam, t) = best
    m = CTWalkQNN(p=d["p"], h=h, t=t, ridge=lam).fit(atr.reshape(-1, 1), ytr)
    pred = m.predict(d["a_te"].reshape(-1, 1))
    # consistency check: dual form on the full training set gives the same predictions
    P = m._probs(atr.reshape(-1, 1)); Pt = m._probs(d["a_te"].reshape(-1, 1))
    pd_ = np.where(dual_fit_predict(P @ P.T, Pt @ P.T, ytr, lam) >= 0, 1, -1)
    assert (pd_ == pred).mean() > 0.99, "dual/primal mismatch"
    return acc, {"h": h, "t": t, "ridge": lam}, pred


def lat_select(d, n):
    g = GridSearchCV(LATKernelSVM(p=d["p"]), {"k": list(range(1, n)), "C": [0.01, 0.1, 1, 10, 100, 1000]},
                     cv=CV, n_jobs=1).fit(d["a_tr"].reshape(-1, 1), d["y_tr"])
    return g.best_score_, {k: (v.item() if hasattr(v, "item") else v) for k, v in g.best_params_.items()}, \
        g.predict(d["a_te"].reshape(-1, 1))


MODELS = {"CTQW-QNN": lambda d, n: ctqw_select(d, n, T_GRID),
          "LAT kernel": lat_select,
          "No walk (t=0)": lambda d, n: ctqw_select(d, n, [0.0])}

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for n in SIZES:
    for draw in DRAWS:
        d = make_lat(n, draw, 150, 100)
        for name, fn in MODELS.items():
            key = f"{n}|{draw}|{name}"
            if key in res:
                continue
            t0 = time.time()
            cv_acc, hp, pred = fn(d, n)
            res[key] = dict(n=n, draw=draw, model=name, cv_acc=float(cv_acc), test_acc=float(np.mean(pred == d["y_te"])),
                            hp=hp, correct=[int(c) for c in pred == d["y_te"]])
            json.dump(res, open(OUT, "w"))
            print(f"n={n:2d} draw={draw} {name:14s} test={res[key]['test_acc']:.3f} {hp} ({time.time() - t0:.1f}s)", flush=True)


def mcn(a, b, sizes):
    ca = np.concatenate([res[f"{n}|{dr}|{a}"]["correct"] for n in sizes for dr in DRAWS])
    cb = np.concatenate([res[f"{n}|{dr}|{b}"]["correct"] for n in sizes for dr in DRAWS])
    x, y = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    return dict(a_only=x, b_only=y, p=float(binomtest(x, x + y, 0.5).pvalue) if x + y else 1.0,
                acc_a=float(ca.mean()), acc_b=float(cb.mean()))


S = {"primary_CTQW_vs_LAT_pooled": mcn("CTQW-QNN", "LAT kernel", SIZES),
     "secondary_CTQW_vs_nowalk_pooled": mcn("CTQW-QNN", "No walk (t=0)", SIZES),
     "per_size": {n: {"vs_LAT": mcn("CTQW-QNN", "LAT kernel", [n]), "vs_nowalk": mcn("CTQW-QNN", "No walk (t=0)", [n])}
                  for n in SIZES},
     "mean_acc": {m: {n: float(np.mean([res[f"{n}|{dr}|{m}"]["test_acc"] for dr in DRAWS])) for n in SIZES} for m in MODELS},
     "sd_acc": {m: {n: float(np.std([res[f"{n}|{dr}|{m}"]["test_acc"] for dr in DRAWS], ddof=1)) for n in SIZES} for m in MODELS}}
p = S["primary_CTQW_vs_LAT_pooled"]
S["decision"] = ("CTQW-QNN outperforms the LAT kernel" if p["p"] < 0.05 and p["a_only"] > p["b_only"] else
                 "LAT kernel outperforms CTQW-QNN" if p["p"] < 0.05 else "Tied: no significant difference")
json.dump(S, open("lat_confirm2_summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
