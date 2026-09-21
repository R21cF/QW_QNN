# implementation_2_pennylane_dataset — a discrete-time quantum walk as a feature map for a quantum kernel

A self-contained kernel experiment on the `qml-benchmarks` dataset generators.
It is exploratory and is not reported in the thesis; the kernel results in the
thesis come from `implementation_1_IBM_dataset/run_kernel.py`.

## What the map is

A coined discrete-time quantum walk on the cycle `Z_{2^n}`: `n` position qubits and one
coin qubit, started localised at `|coin 0>|position 0>`, then `T` steps of

    C(theta_t)   [[cos th,  sin th],
                  [sin th, -cos th]]      real symmetric coin; th = pi/4 is Hadamard
    S            |0>|p> -> |p-1 mod 2^n>,   |1>|p> -> |p+1 mod 2^n>

The kernel is the fidelity `K(x, y) = |<psi(x)|psi(y)>|^2`.

## The design choices, treated as variables

Nothing here is asserted in advance to matter; each is exposed and measured.

| variable | values swept |
|---|---|
| `entry` — where the data enters | `"coin"` (angle `theta_t = theta0 + lam * x_{t mod d}`) or `"position"` (phase layer on the position register) |
| `lam` — encoding scale | 0.02 … 8 |
| `T` — walk steps | 2 … 32 |
| `n` — position qubits, cycle length `2^n` | 2 … 6 |
| `theta0` — coin bias | fixed at pi/4 unless swept |

## Files

| file | what it is |
|---|---|
| `walkmap.py` | the feature map and its kernel; fast exact statevector path used for the hyperparameter search |
| `walkcircuit.py` | the same map as a Qiskit circuit, plus the compute–uncompute fidelity circuit. `check()` asserts circuit and fast path agree — **1.2e-13** over 27 configurations |
| `qkernels.py` | the existing quantum kernels to compare against: Z and ZZ feature maps (Havlicek et al.). `check()` asserts agreement with Qiskit's own `ZFeatureMap` / `ZZFeatureMap` — **1.1e-15** |
| `bench.py` | the comparison. Same splits, same inner CV, tuned grids for every method |
| `probe.py` | the `(lam, T, n)` surface, used to check the optimum is not at a grid boundary |
| `qmlb/` | dataset generators vendored from `XanaduAI/qml-benchmarks` (Apache-2.0, headers intact) — used as a data source only |

Run: `python walkcircuit.py`, `python qkernels.py` (both must print PASS), then `python bench.py`.

## Result

Test accuracy (%), 10 stratified splits, hyperparameters chosen by 3-fold CV on the
training half only, every method on the same splits. Classical baselines get a real
`C`/`gamma` grid; a periodic classical kernel is included because the walk map is
periodic in each feature by construction and omitting it would flatter the walk.

| dataset | walk | Z map | ZZ map | RBF | periodic | linear |
|---|---|---|---|---|---|---|
| two_curves (4f) | 91.4 ± 2.7 | **95.4** | 92.6 | **97.1** | **97.8** | 53.8 |
| lin_sep (4f) | 99.4 ± 1.0 | 98.8 | 98.7 | 98.6 | 91.3 | 99.7 |
| hidden_manifold (4f) | 98.0 ± 1.4 | 98.0 | 97.1 | 97.8 | 97.8 | 96.9 |
| hyperplanes_parity (4f) | 86.8 ± 2.7 | 87.0 | 86.0 | 87.2 | 84.3 | 86.7 |

**The walk kernel does not beat anything.** It ties both existing quantum kernels and the
tuned classical baselines on three datasets, and on the one dataset that discriminates
between methods it is the worst quantum map tested: −4.0 pp against the Z map (p < 0.001)
and −5.7 pp against classical RBF (p < 0.001).

Three things that qualify the table in both directions:

1. **Three of the four datasets are saturated.** On `lin_sep`, `hidden_manifold` and
   `hyperplanes_parity` all six methods sit within about 1.5 pp of each other, so those
   rows cannot distinguish any method from any other. Only `two_curves` carries
   information, and it is negative. Harder instances — more features, fewer samples, more
   noise — are needed before the ties mean anything.
2. **Hyperparameter selection is unstable.** The modal configuration is chosen on only
   2–3 of 10 splits, so the CV is not confidently locating a setting.
3. **The coin-entry variant is usually not selected.** `entry="position"` wins 8–9 of 10
   splits on three datasets and splits 5–5 on the fourth, so letting the data drive the
   walk's trajectory through the coin is, on this evidence, no better than a phase layer
   the walk then redistributes.

## One result that is independent of the comparison

From `probe.py`, on the `(lam, T, n)` surface: the optimum is interior in `lam` and `T`,
not at a grid boundary — on `two_curves` it sits at `lam = 0.5`, `T = 8`, reaching 91.9 %
inner-CV against 51–66 % at the extremes. And **`n` saturates as soon as `2^n` exceeds
roughly `2T`**: `n = 2` is much worse, `n = 4` matches `n = 5` and `n = 6` to within noise.
The cycle only has to be wide enough that the walk does not wrap within `T` steps; extra
position qubits buy nothing after that. That is a statement about the walk's parameters
rather than about the comparison, and it holds regardless of how the comparison turns out.
