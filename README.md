# qwt_c

*Quantum Walks as a Tool for Quantum Machine Learning Algorithms*

Code for the MS thesis. Every figure and table in the thesis is produced by a script in this repository
from a results file that the same directory's simulation scripts write; the map below says which.

## Requirements

Python 3.11+ with numpy, scipy, networkx, scikit-learn, matplotlib, qiskit (>= 2) and qiskit-aer.
`qml-benchmarks` (PennyLane 0.34, JAX 0.4.23) is needed only to re-run the benchmark suite's own
quantum models inside the LAT notebook; nothing the thesis reports depends on it.

## What produces what

| thesis artefact | directory | simulation → results file | report script |
|---|---|---|---|
| `fig_walk_crw.pdf`, `tab_walk_crw.tex` (Ch. 3) | `implementation_8_walk_vs_crw` | `walk_vs_crw.py` → `walk_vs_crw.json` | same script |
| `fig_qaoa_mixers.pdf`, `tab_qaoa.tex` (Ch. 3) | `implementation_3_algorithms` | `qaoa_walk.py` → `qaoa_results.json` | `make_figures.py` |
| `tab_lat_main.tex`, `tab_lat_ablation.tex`, `fig_lat_accuracy.pdf` (Ch. 5) | `implementation_7_qwqnn_lat` | `qwqnn_lat_colab.ipynb` → `lat_results.json` | report cell of the notebook |
| `tab_gradvar_ctqw.tex`, `fig_gradvar_ctqw.pdf` (Ch. 5) | `implementation_7_qwqnn_lat` | `gradvar_ctqw.py` → `gradvar_ctqw.json` | same script |
| `tab_dlp_walk.tex`, `tab_dlp_budget.tex`, `fig_dlp_budget.pdf` (Ch. 5) | `implementation_8_walk_dlp` | `scan_dt_ct.py` → `dlp_scan.json`; `pipeline_walk_dlp.py` → `pipeline_walk_dlp.json` | `report_walk_dlp.py` |
| `tf_text_classification.py`, `torch_image_classification.py` (Appendix) | top level | listings only | — |

The confirmatory runs quoted in Chapter 5 (`lat_confirm*.json`) come from `implementation_7_qwqnn_lat/confirm_run.py`
and `confirm_run2.py`, whose specifications are `PREREGISTRATION.md` and `PREREGISTRATION_2.md`.

The generated `fig_*.pdf` and `tab_*.tex` are copied unchanged into the thesis source directory, which includes
them by file name.

## Shared code

`nb/` holds the LAT data generator, models and selection routines exported from the notebook; the confirmatory
runs and the walk-DLP pipeline `exec` these files. See `nb/README.md` (and run `python nb/export_cells.py --check`
after editing the notebook).

`implementation_3_algorithms` also contains Shor-walk, unstructured-search and graph-learning experiments and
their datasets; these are not reported in the thesis.
