"""LAT pipeline with the lookup table replaced by the walk DLP (DT permutation walk, and the CT variant for comparison).

For every input x of every development draw (n in {8,10,12,16}, draws 0-4, 150 train + 100 test), K_MAX independent
attempts of the walk DLP are simulated (dlp_walk.attempt, at the cost-optimal m from dlp_scan.json). With an attempt
budget K an input gets its exact logarithm if one of its first K attempts verifies, and a uniformly random exponent
otherwise (an unverified output carries no information). K = inf means repeat until verified.
Both quantum classifiers (CTQW-QNN, uncapped; LAT kernel, full grid) are re-selected by CV and trained on the resulting
exponents, then scored on the test set. Classical baselines are unaffected (they never compute the logarithm).
Check: with K = inf the exponents equal log_g x exactly, so predictions must equal the stored lookup-table results.
Writes pipeline_walk_dlp.json."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); NB = os.path.join(HERE, "..", "nb")   # nb/ is exported from the notebook, see nb/README.md
LAT = os.path.join(HERE, "..", "implementation_7_qwqnn_lat")
sys.path.insert(0, HERE)
from dlp_walk import attempt, calls_per_attempt
for f in ["c2_data.py", "c3_models.py"]:
    exec(open(os.path.join(NB, f)).read())
# qml-benchmarks is optional here: only ct_walk_qnn_full and lat_kernel_full are used below
if os.path.isdir(os.path.join(os.environ.get("QMLB_REPO", ""), "src")):
    sys.path.insert(0, os.path.join(os.environ["QMLB_REPO"], "src"))
exec(open(os.path.join(NB, "c4_suite.py")).read())

scan = json.load(open(os.path.join(HERE, "dlp_scan.json")))
BEST_M = {}
for d in scan:
    if d["expected_calls"] is None:
        continue
    key = (d["n"], d["kind"])
    if key not in BEST_M or d["expected_calls"] < BEST_M[key]["expected_calls"]:
        BEST_M[key] = d
K_MAX, BUDGETS = 64, [1, 2, 4, 8, 16, "inf"]
stored = json.load(open(os.path.join(LAT, "lat_results.json")))
OUT = os.path.join(HERE, "pipeline_walk_dlp.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for n in [8, 10, 12, 16]:
    for draw in range(5):
        d = make_lat(n, draw, 150, 100)
        p, g, r = d["p"], d["g"], d["p"] - 1
        a_all = np.r_[d["a_tr"], d["a_te"]]
        for kind in ["DT", "CT"]:
            m = BEST_M[(n, kind)]["m"]
            rng = np.random.default_rng(10_000 * n + 100 * draw + (kind == "CT"))
            first = np.array([np.argmax(ok) if ok.any() else K_MAX
                              for ok in (attempt(kind, int(a), p, g, m, rng, K_MAX) for a in a_all)])
            # repeat-until-verified: inputs without a success in K_MAX attempts keep drawing attempts
            for i in np.nonzero(first == K_MAX)[0]:
                t = K_MAX
                while True:
                    ok = attempt(kind, int(a_all[i]), p, g, m, rng, 256)
                    if ok.any():
                        first[i] = t + int(np.argmax(ok)); break
                    t += 256
                    assert t < 10 ** 6, "input never verifies"
            guess = rng.integers(0, r, len(a_all))
            for K in BUDGETS:
                key = f"{n}|{draw}|{kind}|{K}"
                if key in res:
                    continue
                t0 = time.time()
                if K == "inf":
                    a_hat = a_all.copy(); attempts = first + 1
                else:
                    ok = first < K
                    a_hat = np.where(ok, a_all, guess); attempts = np.minimum(first + 1, K)
                atr, ate = a_hat[:150], a_hat[150:]
                row = dict(n=n, draw=draw, kind=kind, m=m, budget=K,
                           frac_exact=float(np.mean(a_hat == a_all)),
                           mean_attempts=float(attempts.mean()),
                           mean_calls=float(attempts.mean() * calls_per_attempt(kind, m)))
                for name, fn in (("CTQW-QNN", lambda: ct_walk_qnn_full(atr, d["y_tr"], ate, p, n)),
                                 ("LAT kernel", lambda: lat_kernel_full(atr, d["y_tr"], ate, p, n))):
                    cv_acc, hp, pred = fn()
                    row[name] = dict(test_acc=float(np.mean(pred == d["y_te"])), hp={k: (v.item() if hasattr(v, "item") else v) for k, v in hp.items()},
                                     correct=[int(c) for c in pred == d["y_te"]])
                if K == "inf":   # identity check against the lookup-table run
                    for name, sk in (("CTQW-QNN", "QW-QNN (continuous-time walk, full resolution)"), ("LAT kernel", "LAT kernel (full grid)")):
                        ref = stored[f"{n}|{draw}|log x|{sk}"]["correct"]
                        row[name]["identical_to_lookup_table"] = bool(ref == row[name]["correct"])
                res[key] = row
                json.dump(res, open(OUT, "w"))
                print(f"n={n:2d} draw={draw} {kind} m={m} K={K}: exact={row['frac_exact']:.3f} calls/input={row['mean_calls']:.3g} "
                      f"QNN={row['CTQW-QNN']['test_acc']:.3f} LAT={row['LAT kernel']['test_acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
print("done", len(res))
