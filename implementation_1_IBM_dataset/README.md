# implementation_1_IBM_dataset — walk-based classifiers and kernels

Discrete-time coined quantum walks as feature maps and ansätze for a
variational classifier and as state preparation for a fidelity kernel, with
the circuit cost of a walk step measured under several constructions.

## Cores

| file | contents |
|---|---|
| `dtqw.py` | coined DTQW on an arbitrary graph as a Qiskit circuit; Hadamard, Grover, Fourier and Householder coins; per-vertex and per-step coins |
| `fastwalk.py`, `simplewalk.py` | NumPy state-vector implementations of the same walk, used for training |
| `structured_walk.py`, `efficient_walk.py` | edge-coloured (structured) shift construction for cycles, tori and hypercubes |
| `featuremaps.py` | walk feature maps and the reference Z / ZZ maps |
| `qnn_core.py` | the variational models and the training loop |
| `ibm_dataset.py` | the line-detection dataset (2×4 pixel grids) and its splits |

## Checks

`python validate_dtqw.py` and `python crosscheck.py` verify the NumPy cores
against the Qiskit circuits and must pass before anything below is run.
`gauge_experiment.py` tests whether complex coins reach distributions that
real coins cannot.

## Experiments and outputs

| script | writes | then |
|---|---|---|
| `run_qnn.py` | `qnn_results.json` | `plot_qnn.py` → `fig_qnn_training.*`, `fig_qnn_accuracy.*`, `tab_qnn_results.tex` |
| `run_qnn_matched.py` | `qnn_matched.json` | `plot_qnn_matched.py` → `tab_qnn_matched.tex` (real vs complex ansatz coin at matched depth and parameter count; complex arm has its phase frozen at pi/2, three further phases as a robustness check; numpy + scipy only, ~7 min) |
| `run_kernel.py` | `kernel_results.json` | `plot_kernel.py` → `fig_kernel_acc.*`, `tab_kernel.tex` |
| `scaling_structured.py` | `scaling_structured.json` | `scaling_analysis.py` → `coin_cost.json`; `plot_scaling.py` → `fig_walk_scaling.*`, `tab_scaling.tex` |
| `scaling_experiment.py` | — | prints the dense-versus-structured comparison |
| `make_circuits.py` | `fig_circ_walk_fm.*`, `fig_circ_walk_shift.*`, `fig_circ_kernel.*` | circuit diagrams |
| `plot_walk_compare.py` | `walk_compare.json`, `fig_walk_compare.*` | comparison of the classical random walk and the Hadamard walk on the line (exact, NumPy + matplotlib only; independent of `dtqw.py`; not used in the thesis draft) |
| `tune_classical_baselines.py` | `classical_baselines_tuned.json` | classical baselines for the line-detection task (majority class, logistic regression, linear and RBF SVM), hyperparameters chosen by stratified 3-fold CV on the training split, default-parameter scores alongside; seconds |
| `gradient_variance.py` | `gradient_variance.json` | variance of the loss gradient at random parameters against ansatz depth (1–32) on the 5-qubit classifier register: real- and complex-coin walk ansatz vs a hardware-efficient ansatz; finite differences checked against the exact parameter-shift rule; ~5 min |
| `gradient_variance_width.py` | `gradient_variance_width.json` (rows also in `.rows.jsonl`, so an interrupted run resumes) | the same against register width: grid graphs of 8 to 1024 vertices (5–12 qubits), per-vertex vs per-step walk coin vs hardware-efficient ansatz, local and global cost; per-row seeds; ~35 min |
| `plot_gradient_variance.py` | `fig_gradient_variance.*`, `tab_gradvar.tex` | figure and table for the two gradient-variance runs |
| `write_gradvar_section.py` | `sec_gradvar.tex` | generates the thesis subsection text from the two JSON files, so every number in it comes from the recorded runs |

Requirements: numpy, scipy, scikit-learn, matplotlib, qiskit>=2, qiskit-aer.
