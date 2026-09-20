# implementation_3_algorithms — algorithmic use cases and graph-structured data

Simulations behind Chapter 5's §5.1.2–5.1.4 (search, order finding, QAOA
mixers) and §5.4 (graph classification). Added 20 Sep 2026. NumPy state
vectors throughout; Qiskit + Aer where a circuit is the object of the claim
(order finding). Everything is seeded; rerunning reproduces the JSONs.

| script | writes | what it does |
|---|---|---|
| `common.py` | — | coined DTQW on an arbitrary graph as a dense operator (same conventions as implementation_1/dtqw.py); plotting style |
| `search_scaling.py` | `search_results.json` | search on K_N: coined DTQW, CTQW (Childs–Goldstone), nonlinear CTQW (Meyer–Wong); t_peak vs N, FWHM, required g |
| `shor_walk.py` | `shor_results.json`, `shor_circuit_15_7.txt` | U_a as a permutation walk; cycle structure, eigenphases, revival, Qiskit QPE + continued fractions |
| `qaoa_walk.py` | `qaoa_results.json` | balanced Max-Cut, n=8: X+penalty vs CTQW mixer on J(8,4) vs one-step DTQW mixer on J(8,4) |
| `graph_learning.py` | `graph_results.json` | three synthetic graph tasks; DTQW signature kernel, classical RW control, WL, hand-made; walk network with trainable coins |
| `make_figures.py` | `fig_*.pdf/png`, `tab_*.tex` | all figures and tables; copy to ../../thesis/ |

Run order: `python search_scaling.py` (~2 min), `python shor_walk.py` (~4 min),
`python qaoa_walk.py` (~5 min), `python graph_learning.py` (~55 min),
`python make_figures.py`.

Requirements: numpy, scipy, networkx, scikit-learn, matplotlib, qiskit>=2, qiskit-aer.

## Things to know before citing any of this

* **Benchmarks.** `data/MUTAG/` (TU format, from the GraKeL wheel's test data)
  and `data/gin/dataset/{PTC,PROTEINS,IMDBBINARY}/` (GIN txt format, from
  weihua916/powerful-gnns `dataset.zip`). The TU-Dortmund site is unreachable
  from the simulation environment; GitHub's raw CDN was. Published WL numbers
  are reproduced to within noise, which is the protocol check. The quantum
  walk signature does NOT beat the classical walk signature on any benchmark
  (MUTAG: classical ahead by 3 points).
* **Linear walks cannot beat sqrt(N)** (BBBV). The "constant time" in
  `search_scaling.py` is the nonlinear model, and it costs g ~ N^1.27 and a
  peak width ~ N^-1/2. The thesis text says exactly this.
* **Order finding** here is Shor's own QPE on U_a with the controlled powers
  synthesised as dense unitaries — correct, and exponentially expensive in n.
  It is a description of the operator as a walk, not a new algorithm.
* **DTQW QAOA mixer** uses one coined step per layer; at p=1 it is no better
  than a uniform feasible draw. More steps per layer were not tried.
* **Trained coins** in the walk network give no gain over the fixed Grover
  coin (60 COBYLA evaluations per fold).
