# Confirmatory test 2: CTQW-QNN with full resolution vs the LAT quantum kernel

Written 25 Sep 2026, 22:05 PKT. This was fixed before any of the data below was generated or any model was run on it.

## Why a second test
Test 1 (`PREREGISTRATION.md`) gave LAT kernel 34 : 21 CTQW-QNN, p = 0.10. Its result stands and is reported. On review,
test 1 had an asymmetry against the walk model:
- **The walk was capped at resolution h ≤ 6.** The walk ran on the top h ≤ 6 bits of the exponent, a cycle of at most 64 vertices.
  Every exponent in a bin of width (p−1)/64 got the same walk state, so the model could not resolve the interval
  boundaries more finely than that.
- **LAT's kernel had no such cap.** It uses the exact exponent, with widths down to 2^1.

The cap came from the discrete-time model. There, spreading over a wide cycle needs many steps, which costs gates.
For the continuous-time walk that reason does not apply at the times used (t ≤ 8), so the cap is removed.

Extending t down to 0.1 in test 1 was not a handicap: it only added options, since t ∈ {0.25, …, 8} stayed in the grid.

## Changes from test 1
The only change is the walk resolution: **h ∈ {3, 4, …, n}**. At h = n the cycle has 2^n ≥ p−1 vertices, so the resolution is
the full exponent.
- The walk time grid t ∈ {0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8} and the ridge grid {1e-3, 1e-2, 1e-1, 1} are unchanged.
- The kernel width of the walk read-out, about √2·t·(p−1)/2^h exponents, now spans the same range as LAT's widths
  2^k, k ∈ {1, …, n−1}.
- The LAT kernel grid is unchanged from test 1: k ∈ {1, …, n−1}, C ∈ {0.01, 0.1, 1, 10, 100, 1000}.
- The no-walk control (t = 0) gets the same h grid.

The fitting procedure is unchanged: the same ridge problem with an unpenalised bias, solved in dual form when
2^h exceeds the sample count. It is identical to the previous fits; all 20 original draws reproduce exactly.

## Data
Fresh draws, ids 200–209, 10 per size, n ∈ {8, 10, 12, 16}: 40 draws and 4000 test points.
150 training and 100 test inputs per draw, all distinct. CV is 5-fold stratified (random_state 42) on the training set only, and each test set is scored once.

## Hypotheses and tests
- **Primary (one test, α = 0.05, two-sided):** exact McNemar test of CTQW-QNN vs LAT kernel, pooled over all 40 draws.
- **Secondary (reported, not used for the claim):** CTQW-QNN vs no walk (pooled), per-size tests with a Bonferroni correction, and mean accuracies.
- **Decision rule:**
  - "CTQW-QNN outperforms the LAT kernel" only if p < 0.05 with more CTQW-only wins.
  - "LAT kernel outperforms CTQW-QNN" only if p < 0.05 with more LAT-only wins.
  - Otherwise, tied.
- **Tests 1 and 2 are both reported.** Neither replaces the other in the thesis.

## Circuit cost at large h
The check circuit applies e^{-itA} as QFT, an explicit diagonal of 2^h phases, and QFT†. That is exponential in h and is used only for h ≤ 6.
At large h the same unitary needs polynomially many gates in h for bounded t: A = S + S†, where S is the cycle's increment, is 2-sparse.
This can be done by sparse-Hamiltonian simulation, or by computing the phase 2t·cos(2πk/N) arithmetically in the Fourier basis.
In this test, t ≤ 8.

## Not allowed after seeing the results
Changing grids, draws, sizes, the test or the pooling; re-running with other seeds; dropping sizes.
