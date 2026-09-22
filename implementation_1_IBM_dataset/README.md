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
| `run_kernel.py` | `kernel_results.json` | `plot_kernel.py` → `fig_kernel_acc.*`, `tab_kernel.tex` |
| `scaling_structured.py` | `scaling_structured.json` | `scaling_analysis.py` → `coin_cost.json`; `plot_scaling.py` → `fig_walk_scaling.*`, `tab_scaling.tex` |
| `scaling_experiment.py` | — | prints the dense-versus-structured comparison |
| `make_circuits.py` | `fig_circ_walk_fm.*`, `fig_circ_walk_shift.*`, `fig_circ_kernel.*` | circuit diagrams |
| `plot_walk_compare.py` | `walk_compare.json`, `fig_walk_compare.*` | Chapter 2 comparison of the classical random walk and the Hadamard walk on the line (exact, NumPy + matplotlib only; independent of `dtqw.py`) |

Requirements: numpy, scipy, scikit-learn, matplotlib, qiskit>=2, qiskit-aer.
