"""Exact McNemar test on paired per-test-point outcomes (thesis Section 4.4)."""
import numpy as np
from scipy.stats import binomtest


def mcnemar(correct_a, correct_b):
    """(a_only, b_only, p): points only model a got right, only b got right, and the exact two-sided p-value."""
    ca, cb = np.asarray(correct_a), np.asarray(correct_b)
    a_only, b_only = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
    p = float(binomtest(a_only, a_only + b_only, 0.5).pvalue) if a_only + b_only else 1.0
    return a_only, b_only, p
