"""bars_and_stripes under the qml-benchmarks protocol:
GridSearchCV (5-fold, accuracy) over hyperparameters on the training set, then the
best configuration refitted with 5 seeds and scored on the test set.

Arms
  walk-torus : QW-SC, eigenbasis of the CTQW on the L x L torus (the image graph)
  walk-cube  : same, eigenbasis of the CTQW on the hypercube Q_m per register
               (e^{-itA} = product of RX; eigenbasis = Hadamard transform)
  no-walk    : same pipeline, measured in the pixel basis (identity instead of the walk)
  random     : same pipeline, measured in a Haar-random orthogonal basis per register
Finite-shot runs of walk-torus at 1000 and 100 shots per input.
"""
import json
import numpy as np
from scipy.stats import ortho_group
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from data import load
from qwnn import QuantumWalkSpectralClassifier, bin_probs


def hadamard(m):
    H = np.array([[1.0]])
    for _ in range(m):
        H = np.kron(H, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    return H


class BasisVariant(QuantumWalkSpectralClassifier):
    """Replace the walk eigenbasis by another per-register basis; bin by a label map."""

    def __init__(self, C=1.0, shots=None, random_state=42, basis="cube"):
        super().__init__(C=C, shots=shots, random_state=random_state)
        self.basis = basis

    def _features(self, X):
        L = int(round(np.sqrt(X.shape[1]))); m = int(np.log2(L))
        if self.basis == "cube":
            B = hadamard(m)
            lab = np.array([bin(k).count("1") for k in range(L)])   # hypercube eigenvalue m-2|k|
        elif self.basis == "pixel":
            B = np.eye(L); lab = np.arange(L)                          # finest binning: raw |x|^2
        elif self.basis == "random":
            B = ortho_group.rvs(L, random_state=1234); lab = np.arange(L)
        imgs = X.reshape(-1, L, L)
        Y = np.einsum("ij,njk,lk->nil", B, imgs, B)
        P = Y ** 2; P /= P.sum(axis=(1, 2), keepdims=True)
        if self.shots is not None:
            from qwnn import sample_probs
            P = sample_probs(P, self.shots, self.rng_)
        nb = lab.max() + 1
        out = np.zeros((P.shape[0], nb, nb))
        for i in range(L):
            for j in range(L):
                out[:, lab[i], lab[j]] += P[:, i, j]
        return out.reshape(P.shape[0], -1)

    def fit(self, X, y):
        super().fit(X, y)
        return self


GRID = {"C": [1e-2, 1e-1, 1, 10, 100, 1000, 10000]}


def run(make, Xtr, ytr, Xte, yte, seeds=range(5)):
    gs = GridSearchCV(make(0), GRID, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="accuracy", n_jobs=1)
    gs.fit(Xtr, ytr)
    best = gs.best_params_
    rows = []
    for s in seeds:
        mdl = make(s).set_params(**best).fit(Xtr, ytr)
        rows.append((float((mdl.predict(Xtr) == ytr).mean()), float((mdl.predict(Xte) == yte).mean())))
    return best, float(gs.best_score_), rows, mdl


if __name__ == "__main__":
    arms = {
        "walk-torus":        lambda s: QuantumWalkSpectralClassifier(random_state=s),
        "walk-torus 1000sh": lambda s: QuantumWalkSpectralClassifier(shots=1000, random_state=s),
        "walk-torus 100sh":  lambda s: QuantumWalkSpectralClassifier(shots=100, random_state=s),
        "walk-cube":         lambda s: BasisVariant(basis="cube", random_state=s),
        "no-walk (pixel)":   lambda s: BasisVariant(basis="pixel", random_state=s),
        "random basis":      lambda s: BasisVariant(basis="random", random_state=s),
    }
    results = {}
    for L in [4, 8, 16, 32]:
        Xtr, ytr, Xte, yte = load(L)
        for name, mk in arms.items():
            if "sh" in name and L == 32:
                pass
            best, cv, rows, mdl = run(mk, Xtr, ytr, Xte, yte)
            te = np.array([r[1] for r in rows])
            results[f"{name}|{L}"] = {"best": best, "cv": cv, "rows": rows}
            extra = ""
            if name == "walk-torus":
                w = mdl.params_["w"]
                extra = f"  max|w|={np.abs(w).max():.1f}"
            print(f"{L:>2}x{L:<2} {name:<18} C={best['C']:<7g} cv={cv:.3f}  "
                  f"test={te.mean():.3f}±{te.std():.3f} [{te.min():.3f},{te.max():.3f}]{extra}", flush=True)
    json.dump(results, open("results.json", "w"), indent=1)
