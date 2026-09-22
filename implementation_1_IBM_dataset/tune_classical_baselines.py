"""
Classical baselines for the line-detection task, tuned with the same protocol
as the kernel experiments: hyperparameters chosen by stratified 3-fold CV on
the training split only, then scored once on the fixed 140/60 test split.
Untuned (default) scores are printed alongside for comparison.
"""
import json
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from ibm_dataset import load

Xtr, ytr, Xte, yte = load()
C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
cv = StratifiedKFold(3, shuffle=True, random_state=0)
models = {
    "majority class": (DummyClassifier(strategy="most_frequent"), None),
    "logistic regression": (LogisticRegression(max_iter=5000), {"C": C_GRID}),
    "SVM, linear kernel": (SVC(kernel="linear"), {"C": C_GRID}),
    "SVM, RBF kernel": (SVC(kernel="rbf"), {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1, 10]}),
}
out = {}
for name, (est, grid) in models.items():
    default = est.fit(Xtr, ytr).score(Xte, yte)
    if grid is None:
        tuned, best = default, {}
    else:
        gs = GridSearchCV(est, grid, cv=cv).fit(Xtr, ytr)
        tuned, best = gs.score(Xte, yte), gs.best_params_
    out[name] = {"default": default, "tuned": tuned, "best_params": best}
    print(f"{name:22s} default {100*default:5.1f}   tuned {100*tuned:5.1f}   {best}")
json.dump(out, open("classical_baselines_tuned.json", "w"), indent=1, default=str)
