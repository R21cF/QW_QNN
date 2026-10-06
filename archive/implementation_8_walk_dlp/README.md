# implementation_8_walk_dlp — the discrete-logarithm step of the LAT pipeline as a quantum walk

Replaces the lookup table used in `implementation_7_qwqnn_lat` by a simulated walk formulation of Shor's discrete-log
algorithm, compares a discrete-time and a continuous-time formulation, and re-runs the LAT pipeline (walk QNN and LAT kernel)
on the exponents the walk produces.

## Files
| file | what it does |
|---|---|
| `dlp_walk.py` | both formulations: exact outcome-distribution sampler, decoders with classical verification, cost model, `check_against_qiskit` (state-vector simulation of the full circuits at p = 11, 13; agrees to TV distance ~5e-13) |
| `scan_dt_ct.py` → `dlp_scan.json`, `scan.log` | per-attempt success and expected oracle calls vs counting-register size m, n = 8/10/12/16 (25 exponents × 200 attempts) |
| `pipeline_walk_dlp.py` → `pipeline_walk_dlp.json`, `pipeline.log` | the LAT pipeline on walk-computed exponents at attempt budgets K = 1, 2, 4, 8, 16, ∞; both quantum models re-selected by CV per draw; K = ∞ checked to reproduce the lookup-table run exactly (40/40) |
| `report_walk_dlp.py` → `tab_dlp_walk.tex`, `tab_dlp_budget.tex`, `fig_dlp_budget.{pdf,png}`, `pipeline_summary.json` | thesis table/figure and pooled McNemar tests |

Depends on `../nb` (the data generator, models and selectors exported from the LAT notebook, see `../nb/README.md`) and on
`../implementation_7_qwqnn_lat/lat_results.json` for the K = ∞ identity check. qml-benchmarks is not needed.

## Formulations
- **Discrete-time (DT):** phase estimation on the permutation walks U_g: |y> → |gy mod p> and U_x = U_g^a. Both are cyclic
  shifts of the single cycle of length r = p−1 through 1 (g generates), diagonal in the Fourier basis with phases k/r and ka/r.
  Decoding: a = (ka) k^{-1} mod r when gcd(k, r) = 1. This is Shor's DLP algorithm with the eigenphases read as a walk's.
- **Continuous-time (CT):** phase estimation on exp(−iτH), H = U + U†, eigenvalues 2cos(2πk/r); τ = 7π/16 so the spectrum maps
  into (−1/2, 1/2) without aliasing (τ = π/2 aliases ±2 and makes x = −1 undecodable — found and fixed during development).
  The cosine is even in k, so each phase fixes k up to sign; all candidate a are formed.
- Every candidate is verified by g^a = x (one modular exponentiation); an attempt is repeated on failure, so the accepted output is exact.
- **Cost model (oracle calls):** DT: 2m controlled multiplications per attempt (U^{2^j} is one multiplication by g^{2^j}).
  CT: exp(−iτ2^j H) cannot be squared; simulating H for time T needs Ω(T) calls (no fast-forwarding), so ≈ 2τ(2^m − 1) per attempt.

## Result 1: DT is the efficient formulation (`dlp_scan.json`)
| n | DT: best m | success/attempt | E[calls] | CT: best m | success/attempt | E[calls] |
|---|---|---|---|---|---|---|
| 8 | 9 | 0.33 | 55 | 7 | 0.17 | 2.1e3 |
| 10 | 12 | 0.23 | 105 | 9 | 0.11 | 1.2e4 |
| 12 | 14 | 0.27 | 103 | 11 | 0.12 | 4.5e4 |
| 16 | 18 | 0.19 | 185 | 15 | 0.09 | 9.5e5 |

CT costs grow like p (exponential in n); DT costs grow like Shor's. CT saves 4–6 qubits and nothing else. The pipeline uses DT.

## Result 2: the pipeline on walk-computed logarithms (`pipeline_walk_dlp.json`)
- **K = ∞:** every logarithm exact; both models' predictions are identical, test point by test point, to the lookup-table run (40/40).
  Cost 53–194 calls per input. So the earlier QNN/LAT results stand unchanged with a genuine (simulated) walk DLP step.
- **Finite K:** accuracy tracks the fraction of exact logarithms: K = 1 → 52–62 %, K = 8 → 89–98 %, K = 16 within a point of exact.
- **QNN vs LAT kernel at each K (pooled McNemar):** K=1 193:161 p=0.10; **K=2 78:123 p=0.002 (LAT ahead)**; K=4 25:36 p=0.20;
  K=8 10:11 p=1; K=16 12:6 p=0.24; K=∞ 16:2 p=0.001 (development draws — exploratory, see implementation_7).
  With a third to a half of training exponents random, the kernel's local support tolerates the noise better than the ridge read-out.

## Limits
Oracle calls are a query measure, not a gate count (each call is an O(n²)-gate modular multiplication in both formulations).
The outcome distribution, not the circuit, is simulated at n ≥ 8; the full circuits were simulated only at p ≤ 13.
