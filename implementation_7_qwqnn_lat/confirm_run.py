"""Confirmatory run specified in PREREGISTRATION.md. Uses the notebook's data generator and models (c2_data.py, c3_models.py)."""
import json, os, sys, time, warnings
import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
warnings.filterwarnings("ignore")
NB = os.environ.get("NB_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nb"))   # exported from the notebook, see nb/README.md
exec(open(os.path.join(NB, "c2_data.py")).read())
exec(open(os.path.join(NB, "c3_models.py")).read())

CV = StratifiedKFold(5, shuffle=True, random_state=42)
SIZES, DRAWS = [8, 10, 12, 16], list(range(100, 110))
RIDGE = [1e-3, 1e-2, 1e-1, 1.0]
OUT = "lat_confirm.json"


def gs(est, grid, atr, ytr, ate):
    g = GridSearchCV(est, grid, cv=CV, n_jobs=1).fit(atr.reshape(-1, 1), ytr)
    return g.best_score_, {k: (v.item() if hasattr(v, "item") else v) for k, v in g.best_params_.items()}, g.predict(ate.reshape(-1, 1))


MODELS = {
    "CTQW-QNN": lambda d, n: gs(CTWalkQNN(p=d["p"]), {"h": [3, 4, 5, 6], "t": [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8],
                                                      "ridge": RIDGE}, d["a_tr"], d["y_tr"], d["a_te"]),
    "LAT kernel": lambda d, n: gs(LATKernelSVM(p=d["p"]), {"k": list(range(1, n)), "C": [0.01, 0.1, 1, 10, 100, 1000]},
                                  d["a_tr"], d["y_tr"], d["a_te"]),
    "No walk (t=0)": lambda d, n: gs(CTWalkQNN(p=d["p"], t=0.0), {"h": [3, 4, 5, 6], "ridge": RIDGE},
                                     d["a_tr"], d["y_tr"], d["a_te"]),
}

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


summary = {"primary_CTQW_vs_LAT_pooled": mcn("CTQW-QNN", "LAT kernel", SIZES),
           "secondary_CTQW_vs_nowalk_pooled": mcn("CTQW-QNN", "No walk (t=0)", SIZES),
           "per_size": {n: {"vs_LAT": mcn("CTQW-QNN", "LAT kernel", [n]), "vs_nowalk": mcn("CTQW-QNN", "No walk (t=0)", [n])}
                        for n in SIZES},
           "mean_acc": {m: {n: float(np.mean([res[f"{n}|{dr}|{m}"]["test_acc"] for dr in DRAWS])) for n in SIZES} for m in MODELS},
           "sd_acc": {m: {n: float(np.std([res[f"{n}|{dr}|{m}"]["test_acc"] for dr in DRAWS], ddof=1)) for n in SIZES} for m in MODELS}}
p = summary["primary_CTQW_vs_LAT_pooled"]
summary["decision"] = ("CTQW-QNN outperforms the LAT kernel (pre-registered test)"
                       if p["p"] < 0.05 and p["a_only"] > p["b_only"] else
                       "No significant difference: the exploratory 16:3 did not replicate")
json.dump(summary, open("lat_confirm_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
