"""The discrete-logarithm concept class of Liu, Arunachalam and Temme (thesis Sections 3.4 and 5.1.1).

x in Z_p^*, a = log_g x, label y = +1 iff a in [s, s + (p-3)/2] (a half-circle of exponents modulo p-1).
p is the largest n-bit prime and g its smallest generator; draw i of size n is seeded with 1000 n + i
(thesis Section 4.2.3).
"""
import numpy as np

PRIMES = {8: 251, 10: 1021, 12: 4093, 14: 16381, 16: 65521}   # largest n-bit primes


def primitive_root(p):
    """Smallest generator of Z_p^*."""
    phi = p - 1
    fac, m, d = [], phi, 2
    while d * d <= m:
        if m % d == 0:
            fac.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        fac.append(m)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in fac):
            return g


def dlog_table(p, g):
    """log_g for every x in Z_p^*. This brute-force table stands in for Shor's discrete-log
    step in simulation (thesis Section 4.2.2); it is exponential in n classically."""
    log = np.zeros(p, dtype=np.int64)
    v = 1
    for a in range(p - 1):
        log[v] = a
        v = v * g % p
    return log


def bits(v, n):
    """The n bits of each integer in v as +-1 features (the input of the classical comparators)."""
    return (((np.asarray(v)[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)


def make_lat(n, draw, m_train=150, m_test=100):
    """One draw: random offset s, disjoint train/test sets of distinct x."""
    p = PRIMES[n]
    g = primitive_root(p)
    log = dlog_table(p, g)
    rng = np.random.default_rng(1000 * n + draw)
    s = int(rng.integers(p - 1))
    xs = rng.choice(np.arange(1, p), size=m_train + m_test, replace=False)
    a = log[xs]
    y = np.where((a - s) % (p - 1) <= (p - 3) // 2, 1, -1)
    tr, te = slice(0, m_train), slice(m_train, None)
    return dict(n=n, p=p, g=g, s=s, x_tr=xs[tr], x_te=xs[te], a_tr=a[tr], a_te=a[te], y_tr=y[tr], y_te=y[te])
