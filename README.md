# qwt_c

*Quantum Walks as a Tool for Quantum Machine Learning Algorithms* - code for QW-QNN results

## Main result in one notebook

**`ch5_walk_qnn/main_result.ipynb`** regenerates the main result of the thesis (Section 5.1.2): the continuous-time
walk QNN and the Liu–Arunachalam–Temme kernel, both after the discrete-logarithm step, against the classical
baselines (MLP, RBF-SVM, random forest). It writes **Table 5.1** (`results/tab_lat_main.tex`) and **Figure 5.1**
(`results/fig_lat_accuracy.pdf`), prints the McNemar tests quoted in the text, and compares every regenerated
accuracy with the value printed in the thesis. *Run all*; about 5–10 minutes on a laptop CPU.

## Setup

```
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Python 3.11+. PennyLane/JAX are not needed: the benchmark suite's MLP and SVM are scikit-learn models with the
suite's settings, and are built from scikit-learn directly (`qwt/selection.py`).

## Relation to thesis sections/chapters

| thesis | folder | run | output |
|---|---|---|---|
| §3.2 Table 3.1, Fig. 3.1 (walks vs classical random walks) | `ch3_walk_vs_crw` | `walk_vs_crw.py` | `tab_walk_crw.tex`, `fig_walk_crw.pdf`, `walk_vs_crw.json` |
| §3.6.3 Table and figure (QAOA walk mixers) | `ch3_qaoa` | `qaoa_walk.py`, then `report_qaoa.py`; `check_qiskit.py` checks the mixers against Qiskit circuits | `qaoa_results.json`, `tab_qaoa.tex`, `fig_qaoa_mixers.pdf` |
| §5.1.2 Table 5.1, Fig. 5.1 (**main result**) | `ch5_walk_qnn` | `main_result.ipynb` | `results/` |
| §5.1.3 DLP step as a walk | `ch5_walk_dlp` | `dlp_walk.py` (Qiskit check), `scan_dt_ct.py`, `pipeline_walk_dlp.py` (needs the notebook's `results/main_result.json`), `report_walk_dlp.py` | `dlp_scan.json`, `pipeline_walk_dlp.json`, `tab_dlp_walk.tex`, `tab_dlp_budget.tex`, `fig_dlp_budget.pdf` |
| §5.1.4 gradient variance | `ch5_gradvar` | `gradvar_ctqw.py` | `gradvar_ctqw.json`, `tab_gradvar_ctqw.tex`, `fig_gradvar_ctqw.pdf` |
| Appendix A listings | `appendix_a` | — (the listings printed in the thesis) | — |


## Shared code: `qwt/`

`lat.py` the discrete-logarithm concept class · `walks.py` the walks on the cycle and their Qiskit circuits ·
`models.py` the walk QNN and the LAT kernel · `selection.py` the cross-validation protocol and grids ·
`stats.py` the exact McNemar test.

## `archive/`

Everything not used by `draft_e.tex` (earlier benchmark experiments, QRNG, graph learning, search, the Colab
notebook this code replaces). Kept on disk and ignored by git.
