"""Per-attempt success and expected oracle calls of the DT and CT walk DLP, n = 8..16, as a function of the
counting-register size m. Writes dlp_scan.json."""
import json, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from dlp_walk import attempt, calls_per_attempt
from qwt.lat import PRIMES, primitive_root
os.chdir(HERE)

N_A, N_ATT = 25, 200          # 25 random exponents x 200 attempts per (kind, n, m)
out = []
for n in [8, 10, 12, 16]:
    p = PRIMES[n]; g = primitive_root(p); r = p - 1
    rng = np.random.default_rng(n)
    avals = rng.integers(0, r, N_A)
    for kind, ms in (("DT", range(max(2, n - 3), n + 6)), ("CT", range(max(2, n - 4), 2 * n + 7))):
        for m in ms:
            succ = np.mean([attempt(kind, int(a), p, g, m, np.random.default_rng(1000 * m + i), N_ATT).mean()
                            for i, a in enumerate(avals)])
            calls = calls_per_attempt(kind, m)
            row = dict(n=n, p=p, kind=kind, m=m, success=float(succ), calls_per_attempt=calls,
                       expected_calls=float(calls / succ) if succ > 0 else None,
                       qubits=int(np.ceil(np.log2(p))) + 2 * m)
            out.append(row)
            print(row, flush=True)
json.dump(out, open("dlp_scan.json", "w"), indent=1)
