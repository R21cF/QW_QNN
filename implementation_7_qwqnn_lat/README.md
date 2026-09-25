# implementation_7_qwqnn_lat — Hadamard-coin quantum-walk QNN on the discrete-log concept class

`qwqnn_lat_colab.ipynb` is self-contained. It regenerates the Liu–Arunachalam–Temme (LAT, *Nat. Phys.* 2021)
discrete-log dataset, defines the QW-QNN and its comparators, runs the protocol, and writes the tables and figure.
`lat_results.json` holds every fit. The notebook skips finished fits, so copying this file into
`MyDrive/qw_qnn_lat/` before running on Colab only regenerates the tables and figure.

## Model (QW-QNN)
The structure comes from IBM's QVC/QNN lesson: feature map → trainable part → expectation value → MSE.
1. **Discrete-log encoding:** |x⟩ → |a = log_g x⟩. This is Shor's DLP subroutine, as in LAT; in simulation it is computed classically.
2. **Walk:** a coin qubit starts in (|0⟩+i|1⟩)/√2, and an h-qubit position register holds the top h bits of a (a vertex of the cycle C_{2^h}).
   The walk runs t steps of W = S(H ⊗ I), where S = |0⟩⟨0|⊗INC + |1⟩⟨1|⊗DEC.
3. **Readout:** measure the position and output f(a) = Σ_u w_u P_t(u|a) + b, the expectation of a trainable diagonal observable.
   The loss is MSE on y = ±1. Because it is linear in (w, b), it is minimised exactly (ridge).

The walk circuit agrees with Qiskit's `Statevector` to 8.5e-14 (notebook cell `c3b`).

## Data and protocol
- **Data:** p is the largest n-bit prime and g its smallest generator. The label is y = +1 iff log_g x ∈ [s, s+(p−3)/2], with s random per draw.
- **Samples:** 150 train and 100 test, all distinct x, so train and test never share an x. There are 5 draws for each n ∈ {8, 10, 12, 16}.
- **Selection and scoring:** 5-fold CV on the training set only, then each test set is scored once. Grids use the qml-benchmarks values for C, SVC and MLP.
- **Deviation from the suite:** the data re-uploading grid is reduced from 36 settings to 8.
- **Inputs:** every model receives the same input, x (as ±1 bits for the classical and suite models).
  The quantum pipelines (QW-QNN, LAT kernel) apply Shor's discrete-log circuit to x as their first layer; in simulation
  that step is a classical lookup table. `lat_results.json` tags these records `input: "log x"` to mark where the step was
  applied, not to indicate a different input.

## Results (mean test accuracy ± s.d. over 5 draws)

### Main comparison: every model receives x

| model | n=8 | n=10 | n=12 | n=16 |
|---|---|---|---|---|
| **QW-QNN: Shor DLP → Hadamard walk → readout** | **0.990 ± 0.014** | 0.990 ± 0.010 | **0.998 ± 0.004** | **0.986 ± 0.017** |
| LAT: Shor DLP → quantum kernel SVM | 0.984 ± 0.011 | **0.996 ± 0.005** | 0.988 ± 0.018 | 0.980 ± 0.023 |
| MLP | 0.496 ± 0.046 | 0.498 ± 0.026 | 0.494 ± 0.032 | 0.488 ± 0.057 |
| SVM (RBF) | 0.488 ± 0.042 | 0.484 ± 0.038 | 0.474 ± 0.053 | 0.542 ± 0.064 |
| Random forest | 0.500 ± 0.023 | 0.486 ± 0.062 | 0.436 ± 0.035 | 0.502 ± 0.082 |

### Ablations: what the discrete-log step and the walk each contribute

| model | n=8 | n=10 | n=12 | n=16 |
|---|---|---|---|---|
| Shor DLP → no walk (t = 0) → readout | 0.970 ± 0.019 | 0.992 ± 0.008 | 0.986 ± 0.026 | 0.976 ± 0.021 |
| log x given (not classically efficient) → MLP | 0.976 ± 0.013 | 0.946 ± 0.042 | 0.938 ± 0.056 | 0.900 ± 0.054 |
| log x given (not classically efficient) → SVM (RBF) | 0.968 ± 0.015 | 0.926 ± 0.027 | 0.854 ± 0.036 | 0.844 ± 0.099 |
| log x given (not classically efficient) → Random forest | 0.928 ± 0.050 | 0.942 ± 0.043 | 0.890 ± 0.025 | 0.884 ± 0.071 |

The first ablation removes the walk but keeps the discrete-log step. The other three give classical models the output
of the discrete-log step directly, which no efficient classical algorithm can compute at large n. They measure how much
of the result is the discrete-log step alone.

With finite measurement shots the QW-QNN scores 0.990 / 0.990 / 0.996 / 0.978 (100 shots) and 0.990 / 0.990 / 0.998 / 0.984 (1000 shots).

**Paired exact McNemar tests** compare the QW-QNN with each comparator on test points pooled over the 5 draws.
The counts are test points only the QW-QNN got right vs. points only the comparator got right.
- **vs. the best classical model:** 249:4, 249:3, 252:0, 225:3, all p < 1e-60.
- **vs. the LAT kernel:** 3:0, 1:4, 6:1, 5:2, with p = 0.25, 0.38, 0.13, 0.45. None is significant.
- **vs. the no-walk control:** 10:0, 1:2, 7:1, 5:0, with p = 0.002, 1.0, 0.07, 0.06.
  Only n = 8 survives a Bonferroni correction over the 4 sizes (threshold 0.0125).

## What these results support, and what they don't
1. **Beats the classical baselines: yes, but through the discrete-log step, not the walk.** Every classical model sits at chance.
   Classical models handed the discrete log reach 0.84–0.98 (ablations), so the entire gap is the Shor step. The gap is also only asymptotic:
   at n ≤ 16 a classical brute-force discrete log is trivial.
2. **Best quantum model: not shown.** The QW-QNN has the highest mean at 3 of 4 sizes, but it is statistically tied with LAT's
   kernel at every size. Every pipeline with the discrete-log step is near the ceiling, set by the boundary bins of the interval, so this dataset
   has almost no room to separate quantum models. The comparison is therefore against LAT's kernel only; the
   qml-benchmarks quantum models were not run (see below).
3. **Does the walk help over no walk (first ablation)?** Weakly. The QW-QNN is ahead at 3 of 4 sizes and significant only at n = 8.
   CV often picks t = 1 (11 of 20 draws), where the walk just averages neighbouring bins.
4. **Scalability:** the walk runs on the top h ≤ 6 bits of the exponent (a cycle of at most 64 vertices).
   A walk on the full exponent cycle would need about p/m steps, which is exponential in n.

## qml-benchmarks quantum models: stopped
The IQP kernel, projected quantum kernel and data re-uploading classifier were stopped after 5 fits, all at n = 8, draw 0
(about 3–13 min per fit at 8 qubits, hours per size at 10–12). Those fits stay in `lat_results.json` but are excluded
from the tables and figure because they cover a single draw. Test accuracy on that draw was:
- **On x directly:** IQP 0.44, projected kernel 0.39, data re-uploading 0.47.
- **After the Shor DLP step:** IQP 0.74, projected kernel 0.96.
- **For comparison, on the same draw:** QW-QNN 0.97 and LAT kernel 0.97.

Because these models weren't run, no claim can be made against them, including "best quantum model" beyond LAT's kernel.
To run them, set `SKIP_MODELS = set()` in the config cell.

## Files
- `qwqnn_lat_colab.ipynb`: the full experiment. Set `SMOKE = True` for a roughly one-minute check. It was executed end to end locally in smoke mode.
- `lat_results.json`: per-fit records (CV accuracy, chosen hyperparameters, per-test-point correctness, shot results).
- `fig_lat_accuracy.{png,pdf}`, `lat_summary.csv` (main), `lat_ablation.csv`, `lat_mcnemar.csv`: generated by the report cell.

## Run on Colab
Upload the notebook and put `lat_results.json` in `MyDrive/qw_qnn_lat/`. Then *Runtime → Run all*.
After the pinned-package install (PennyLane 0.34 and JAX 0.4.23, the versions qml-benchmarks needs), choose
*Runtime → Restart session* and *Run all* again. A CPU runtime is enough. The pinned install was tested in a
Python 3.11 venv, not on Colab itself.
