# implementation_3_algorithms — algorithmic use cases and graph-structured data

Simulations of unstructured search, order finding and QAOA mixers, and the
graph-classification experiment. NumPy state vectors throughout; Qiskit + Aer
where a circuit is the object of study (order finding). Everything is seeded;
rerunning reproduces the JSON result files.

| script | writes | what it does |
|---|---|---|
| `common.py` | — | coined DTQW on an arbitrary graph as a dense operator (same conventions as `implementation_1_IBM_dataset/dtqw.py`); plotting style |
| `search_scaling.py` | `search_results.json` | search on K_N: coined DTQW, CTQW (Childs–Goldstone), nonlinear CTQW (Meyer–Wong); time to first peak vs N, peak width, required nonlinearity |
| `shor_walk.py` | `shor_results.json`, `shor_circuit_15_7.txt` | U_a as a permutation walk; cycle structure, eigenphases, revival, Qiskit QPE + continued fractions |
| `qaoa_walk.py` | `qaoa_results.json` | balanced Max-Cut, n=8: X mixer + penalty vs CTQW mixer on J(8,4) vs one-step DTQW mixer on J(8,4) |
| `graph_learning.py` | `graph_results.json` | three synthetic graph tasks and four public benchmarks; DTQW signature kernel, classical random-walk control, WL, hand-made statistics; walk network with trainable coins |
| `make_figures.py` | `fig_*.pdf/png`, `tab_*.tex` | all figures and tables |

Run order: `python search_scaling.py` (~2 min), `python shor_walk.py` (~4 min),
`python qaoa_walk.py` (~5 min), `python graph_learning.py` (~55 min),
`python make_figures.py`.

Requirements: numpy, scipy, networkx, scikit-learn, matplotlib, qiskit>=2, qiskit-aer.

## Notes on the data and the results

* `data/MUTAG/` is in TU format (copy distributed with GraKeL);
  `data/gin/dataset/{PTC,PROTEINS,IMDBBINARY}/` are in the GIN text format
  (from the `powerful-gnns` dataset bundle). Published WL numbers are
  reproduced to within noise, which is the protocol check.
* The quantum walk signature does not beat the classical walk signature on
  any benchmark (MUTAG: classical ahead by 3 points). The quantum start state
  (uniform superposition over arcs) is a fixed point of the Grover/flip-flop
  walk for every coin angle, so the entropy, participation and maximum
  features are constant in t; only the return-probability features carry
  dynamics. A rerun from a non-stationary start state is pending.
* Linear walks cannot beat sqrt(N) (BBBV). The constant-time result in
  `search_scaling.py` is the nonlinear model; it costs g ~ N^1.27 and a peak
  width ~ N^-1/2.
* Order finding is Shor's QPE on U_a with the controlled powers synthesised
  as dense unitaries: correct, and exponentially expensive in n. It is a
  description of the operator as a walk, not a new algorithm.
* The DTQW QAOA mixer uses one coined step per layer; at p=1 it is no better
  than a uniform feasible draw. More steps per layer were not tried.
* Trained coins in the walk network give no gain over the fixed Grover coin
  (60 COBYLA evaluations per fold).
