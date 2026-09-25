"""Thesis outputs for the quantum-walk kernel on hyperplanes_diff (qml-benchmarks).

Re-runs the pre-planned protocol of final_kernel.py (run 1) with per-test-point predictions kept, checks it against
kernel_results.csv, compares with the suite's published results, and writes
    tab_kernel_hp.tex, fig_kernel_hp.{pdf,png}, kernel_hp_stats.json
Requires the qml-benchmarks repository (for the published results) at QMLB_REPO (default ./qml-benchmarks).
"""
import os, re, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from data import load_all
from walkkernel import walk_kernel

REPO = os.environ.get("QMLB_REPO", "qml-benchmarks")
RES = os.path.join(REPO, "paper/results/hyperplanes_diff")
D = load_all()
cv = StratifiedKFold(5, shuffle=True, random_state=42)
Cs = [0.3, 1, 3, 10, 30, 100, 300]
ARMS = {"walk": [("walk", s, n) for n in [8, 16, 64] for s in [0.05, 0.1, 0.2, 0.4, 0.8]],
        "rbf": [("rbf", g) for g in [0.003, 0.01, 0.03, 0.1, 0.3, 1]]}


def K_of(sp, A, B):
    return rbf_kernel(A, B, gamma=sp[1]) if sp[0] == "rbf" else walk_kernel(A, B, sp[1], sp[2], "product")


# 1. data check: the suite's SVC, refitted with its published best hyperparameters, must reproduce its test accuracy
ok = 0
for k in range(2, 21):
    Xtr, ytr, Xte, yte = D[k]
    hp = pd.read_csv(f"{RES}/SVC/SVC_hyperplanes-10d-from3d-{k}n_GridSearchCV-best-hyperparams.csv")
    hp = dict(zip(hp.iloc[:, 0], hp.iloc[:, 1]))
    pub = pd.read_csv(f"{RES}/SVC/SVC_hyperplanes-10d-from3d-{k}n_GridSearchCV-best-hyperparams-results.csv").test_acc.mean()
    acc = (SVC(C=float(hp["C"]), gamma=float(hp["gamma"])).fit(Xtr, ytr).predict(Xte) == yte).mean()
    ok += abs(acc - pub) < 1e-9
print(f"data check: published SVC test accuracy reproduced on {ok}/19 instances")

# 2. run-1 protocol with predictions kept
rows, correct = [], {}
for arm, specs in ARMS.items():
    for k in range(2, 21):
        Xtr, ytr, Xte, yte = D[k]
        best = (-1, None, None)
        for sp in specs:
            K = K_of(sp, Xtr, Xtr)
            for C in Cs:
                acc = np.mean([(SVC(kernel="precomputed", C=C).fit(K[np.ix_(i, i)], ytr[i])
                                .predict(K[np.ix_(j, i)]) == ytr[j]).mean() for i, j in cv.split(Xtr, ytr)])
                if acc > best[0]:
                    best = (acc, sp, C)
        cvacc, sp, C = best
        pred = SVC(kernel="precomputed", C=C).fit(K_of(sp, Xtr, Xtr), ytr).predict(K_of(sp, Xte, Xtr))
        correct[(arm, k)] = (pred == yte).astype(int)
        rows.append(dict(arm=arm, k=k, cv=cvacc, test=(pred == yte).mean(), spec=str(sp), C=C))
ours = pd.DataFrame(rows)
old = pd.read_csv("kernel_results.csv")
old_w = old[old.arm.str.startswith("QW")].test.values; old_r = old[old.arm.str.startswith("RBF")].test.values
assert np.allclose(ours[ours.arm == "walk"].test.values, old_w) and np.allclose(ours[ours.arm == "rbf"].test.values, old_r), \
    "re-run differs from kernel_results.csv"
print("re-run matches kernel_results.csv on all 38 rows")

# 3. published results
pub = []
for f in glob.glob(f"{RES}/*/*-best-hyperparams-results.csv"):
    m = f.split(os.sep)[-2]; k = int(re.search(r"from3d-(\d+)n", f).group(1))
    pub.append(dict(model=m, k=k, test=pd.read_csv(f).test_acc.mean()))
pub = pd.DataFrame(pub)
per = pub.pivot(index="k", columns="model", values="test")
per["QW product kernel"] = ours[ours.arm == "walk"].set_index("k").test
per["RBF-SVM, same protocol"] = ours[ours.arm == "rbf"].set_index("k").test
means = per.mean().sort_values(ascending=False)
print(means.round(4).to_string())

# 4. paired tests over the 19 instances (published models: per-instance mean over the suite's seeds)
walk = per["QW product kernel"]
stats = {"mean_test": means.round(4).to_dict(),
         "data_check_svc_reproduced": int(ok)}
for other in ["MLPClassifier", "DressedQuantumCircuitClassifier", "DataReuploadingClassifier",
              "IQPKernelClassifier", "ProjectedQuantumKernel", "SVC", "RBF-SVM, same protocol"]:
    d = walk - per[other]
    nz = d[np.abs(d) > 1e-12]
    stats[f"walk_vs_{other}"] = dict(mean_diff=round(float(d.mean()), 4), wins=int((d > 1e-12).sum()),
                                     losses=int((d < -1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                                     wilcoxon_p=round(float(wilcoxon(nz).pvalue), 4) if len(nz) else 1.0)
cw = np.concatenate([correct[("walk", k)] for k in range(2, 21)])
cr = np.concatenate([correct[("rbf", k)] for k in range(2, 21)])
a, b = int(((cw == 1) & (cr == 0)).sum()), int(((cw == 0) & (cr == 1)).sum())
stats["mcnemar_walk_vs_rbf_pooled"] = dict(walk_only=a, rbf_only=b, p=round(float(binomtest(a, a + b, 0.5).pvalue), 4))
# closeness to an RBF kernel: |G_n(t)|^2 ~ exp(-2 t^2) for small t, so the product kernel ~ RBF with gamma = 2 s^2
cl = []
for _, r in ours[ours.arm == "walk"].iterrows():
    sp = eval(r.spec); Xtr = D[int(r.k)][0]
    Kw = walk_kernel(Xtr, Xtr, sp[1], sp[2], "product"); Kr = rbf_kernel(Xtr, Xtr, gamma=2 * sp[1] ** 2)
    off = ~np.eye(len(Xtr), dtype=bool)
    cl.append((np.abs(Kw - Kr)[off].max(), np.corrcoef(Kw[off], Kr[off])[0, 1]))
cl = np.array(cl)
stats["walk_vs_rbf_gamma_2s2"] = dict(max_abs_diff=round(float(cl[:, 0].max()), 4), min_corr=round(float(cl[:, 1].min()), 4))
stats["walk_selected_n_vertices"] = ours[ours.arm == "walk"].spec.str.extract(r", (\d+)\)")[0].value_counts().to_dict()
json.dump(stats, open("kernel_hp_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

# 5. table
SHOW = [("QW product kernel", r"Walk product kernel (this work)", "quantum"),
        ("DressedQuantumCircuitClassifier", "Dressed quantum circuit", "quantum"),
        ("DataReuploadingClassifier", "Data re-uploading", "quantum"),
        ("IQPKernelClassifier", "IQP kernel", "quantum"),
        ("ProjectedQuantumKernel", "Projected quantum kernel", "quantum"),
        ("MLPClassifier", "Multilayer perceptron", "classical"),
        ("RBF-SVM, same protocol", r"RBF-kernel SVM, this protocol", "classical"),
        ("SVC", "RBF-kernel SVM, suite protocol", "classical")]
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption[Hyperplanes task: walk kernel against the benchmark suite]{Test accuracy (\%) on the nineteen "
     r"\texttt{hyperplanes\_diff} instances of the benchmark suite~\cite{Bowles_2024}: mean over instances, and "
     r"wins/losses/ties of the walk kernel per instance. Suite models are its published results, each averaged over "
     r"its five seeds; the two rows marked \emph{this protocol} and \emph{this work} are computed here under an identical "
     r"protocol.}", r"\label{tab:kernelhp}", r"\begin{tabular}{llrr}", r"\hline",
     r"Model & Type & Mean acc.\ (\%) & Walk W/L/T \\", r"\hline"]
for key, name, typ in SHOW:
    s = stats.get(f"walk_vs_{key}")
    wlt = "--" if s is None else f"{s['wins']}/{s['losses']}/{s['ties']}"
    L.append(f"{name} & {typ} & ${100 * means[key]:.1f}$ & {wlt} \\\\")
L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
open("tab_kernel_hp.tex", "w").write("\n".join(L))

# 6. figure: per-instance test accuracy against the number of hyperplanes
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
for key, lab, col, mk, ls in [("QW product kernel", "Walk product kernel", "#2a78d6", "o", "-"),
                              ("MLPClassifier", "Multilayer perceptron (suite)", "#eb6834", "s", "-"),
                              ("DressedQuantumCircuitClassifier", "Best suite quantum model (dressed circuit)", "#1baf7a", "^", "-"),
                              ("RBF-SVM, same protocol", "RBF-kernel SVM, same protocol", "#eda100", "D", "--")]:
    ax.plot(per.index, per[key], color=col, marker=mk, ms=4.5, lw=1.8, ls=ls, label=f"{lab} ({100 * means[key]:.1f}%)")
ax.set_xlabel("number of hyperplanes $k$"); ax.set_ylabel("test accuracy")
ax.set_xticks(range(2, 21, 2)); ax.set_ylim(0.5, 1.0)
ax.grid(axis="y", color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig("fig_kernel_hp.pdf"); fig.savefig("fig_kernel_hp.png")
print("wrote tab_kernel_hp.tex, fig_kernel_hp.pdf/png, kernel_hp_stats.json")
