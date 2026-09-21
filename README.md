# qwt_code — simulations for *Quantum Walks as a Tool for Quantum Machine Learning Algorithms*

Simulation code, seeds, recorded results and the figure- and table-generating
scripts for the MS thesis of that title (School of Natural Sciences, NUST).

| directory | contents |
|---|---|
| `implementation_1_IBM_dataset/` | walk-based variational classifiers and fidelity kernels; circuit cost of a walk step |
| `implementation_3_algorithms/` | unstructured search, order finding and QAOA mixers with walks; graph classification with walk signatures |
| `implementation_2_pennylane_dataset/` | an exploratory kernel experiment on the `qml-benchmarks` generators (not reported in the thesis) |
| `QRNG/` | quantum-walk random-number generation notebooks (side project) |

Each directory has its own README with the run order. Every experiment is
seeded and writes its results to a JSON file; the plotting scripts read those
files, so the tables and figures can be regenerated without rerunning the
simulations.

Python 3.11+, numpy, scipy, networkx, scikit-learn, matplotlib, qiskit>=2,
qiskit-aer; `implementation_2_pennylane_dataset` additionally uses the vendored
`qmlb` generators. The two top-level classical listings
(`tf_text_classification.py`, `torch_image_classification.py`) are the
reference examples reproduced in the thesis appendix.
