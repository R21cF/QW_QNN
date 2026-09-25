"""Exact reproduction of the qml-benchmarks bars_and_stripes datasets
(paper/benchmarks/generate_bars_and_stripes.py), plus the Bayes-optimal
classifier, which fixes the ceiling any model can reach on the fixed test set."""
import numpy as np
from bars_and_stripes import generate_bars_and_stripes

NOISE = 0.5
SIZES = [4, 8, 16, 32]


def load(size):
    np.random.seed(42)  # same call order as the suite's script
    Xtr, ytr = generate_bars_and_stripes(1000, size, size, NOISE)
    Xte, yte = generate_bars_and_stripes(200, size, size, NOISE)
    return Xtr.reshape(1000, -1), ytr, Xte.reshape(200, -1), yte


def _log_mix(v, s=NOISE):
    # log( 0.5 N(v;+1,s) + 0.5 N(v;-1,s) ) up to a constant, summed over axis 0
    a = -((v - 1) ** 2) / (2 * s * s)
    b = -((v + 1) ** 2) / (2 * s * s)
    return np.logaddexp(a, b).sum(axis=0)


def bayes_llr(X, size):
    """log p(x|stripe, y=+1) - log p(x|bar, y=-1).  Stripe: rows constant;
    bar: columns constant.  Each row/column independently on/off with prob 1/2."""
    out = []
    for x in X.reshape(-1, size, size):
        # stripe: for each row r, all pixels share one value in {+1,-1}
        a_r = -((x - 1) ** 2).sum(1) / (2 * NOISE**2)
        b_r = -((x + 1) ** 2).sum(1) / (2 * NOISE**2)
        ls = np.logaddexp(a_r, b_r).sum()
        a_c = -((x - 1) ** 2).sum(0) / (2 * NOISE**2)
        b_c = -((x + 1) ** 2).sum(0) / (2 * NOISE**2)
        lb = np.logaddexp(a_c, b_c).sum()
        out.append(ls - lb)
    return np.array(out)


if __name__ == "__main__":
    for L in SIZES:
        Xtr, ytr, Xte, yte = load(L)
        llr = bayes_llr(Xte, L)
        pred = np.where(llr >= 0, 1.0, -1.0)
        # ambiguous test images: noise-free image all +1 or all -1
        # (llr exactly 0 up to float); count by label disagreement risk
        amb = np.sum(np.abs(llr) < 1e-9)
        print(f"{L}x{L}: Bayes-optimal test acc = {np.mean(pred == yte):.3f}"
              f"  (train {np.mean(np.where(bayes_llr(Xtr, L) >= 0, 1, -1) == ytr):.3f});"
              f" exact ties={amb}")
