# Confirmatory test: continuous-time-walk QNN vs the LAT quantum kernel

Written 25 Sep 2026, 21:50 PKT. This was fixed before any of the data below was generated or any model was run on it.

## Why this test
The exploratory result was a pooled McNemar test over the 20 original draws (draw ids 0–4): CTQW-QNN 16 : 3 LAT kernel, p = 0.004.
That comparison was chosen after the per-size results had been seen, and the LAT kernel's width grid was narrow
(k ∈ {n−6, …, n−2}). This run checks whether the difference survives fresh data and a fairly tuned LAT kernel.

## Data
- The same LAT concept class and generator as before (`make_lat`).
- Sizes: n ∈ {8, 10, 12, 16}.
- Draws: **new draw ids 100–109**, 10 per size, 40 in total, none shared with the exploratory run.
- 150 training and 100 test inputs per draw, all distinct.

## Models, grids and protocol
Hyperparameters are chosen by 5-fold stratified CV (random_state 42) on the training set only. Each test set is scored once.
- **CTQW-QNN:** h ∈ {3, 4, 5, 6}, t ∈ {0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8}, ridge ∈ {1e-3, 1e-2, 1e-1, 1}.
  The grid is extended below 0.25 because CV chose the grid edge before.
- **LAT kernel SVM:** k ∈ {1, …, n−1} (every interval width 2^k), C ∈ {0.01, 0.1, 1, 10, 100, 1000}.
  Both grids are widened so neither model is disadvantaged by its grid.
- **No-walk control:** the CTQW-QNN at t = 0, with h and ridge grids as above.

## Hypotheses and tests
- **Primary (one test, α = 0.05, two-sided):** exact McNemar test of CTQW-QNN vs LAT kernel,
  on test points pooled over all 40 draws (4000 test points).
- **Secondary (reported, not used for the claim):**
  - CTQW-QNN vs no-walk control, same pooled test.
  - Per-size McNemar tests, with a Bonferroni correction over the 4 sizes.
  - Mean accuracies.
- **Decision rule:**
  - "CTQW-QNN outperforms the LAT kernel on this task" is stated in the thesis only if the primary test gives p < 0.05
    with more CTQW-only wins than LAT-only wins.
  - Otherwise the thesis states that the two are statistically tied, and the exploratory 16 : 3 is reported as not replicated.

## Not allowed after seeing the results
Changing grids, draws, sizes, the test, or the pooling. Re-running with other seeds. Dropping sizes.
