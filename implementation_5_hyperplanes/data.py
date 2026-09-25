"""Exact regeneration of qml-benchmarks hyperplanes_diff (paper/benchmarks/generate_hyperplanes.py)."""
import numpy as np
from sklearn.model_selection import train_test_split
from hyperplanes import generate_hyperplanes_parity

def load_all():
    np.random.seed(1)
    out = {}
    for k in range(2, 21):
        X, y = generate_hyperplanes_parity(300, 10, k, 3)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)
        out[k] = (Xtr, ytr, Xte, yte)
    return out
