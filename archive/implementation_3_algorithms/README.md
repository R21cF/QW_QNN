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
| `graph_learning.py` | `graph_results.json` (rows also in `graph_results.rows.jsonl`, so an interrupted run resumes) | three synthetic graph tasks and four public benchmarks; DTQW signature kernel, classical random-walk control, WL, hand-made statistics; walk network with trainable coins |
| `make_figures.py` | `fig_*.pdf/png`, `tab_*.tex` | all figures and tables |

Run order: `python search_scaling.py` (~2 min), `python shor_walk.py` (~4 min),
`python qaoa_walk.py` (~5 min), `python graph_learning.py` (~1 h on two cores),
`python make_figures.py`.

Requirements: numpy, scipy, networkx, scikit-learn, matplotlib, qiskit>=2, qiskit-aer.

## Notes on the data and the results

* `data/MUTAG/` is in TU format (copy distributed with GraKeL);
  `data/gin/dataset/{PTC,PROTEINS,IMDBBINARY}/` are in the GIN text format
  (from the `powerful-gnns` dataset bundle). Published WL numbers are
  reproduced to within noise, which is the protocol check.
* Both walk signatures start a walk at every vertex in turn (the quantum walk
  in the uniform superposition over that vertex's arcs) and average the
  entropy, participation and maximum of the position marginal over the start
  vertex; the classical control is the lazy random walk under the same
  protocol. The uniform superposition over *all* arcs is not used as a start:
  it is a fixed point of the Grover/flip-flop walk for every coin angle. Each
  dataset has its own seed, so results do not depend on run order.
* Under this protocol the quantum and classical walk signatures are within
  noise of each other on all four benchmarks. The quantum walk's margin on the
  bipartite task is plausibly due to the laziness of the classical control
  (a non-lazy control was not run).
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
