# qwt_c

*Quantum Walks as a Tool for Quantum Machine Learning Algorithms* - code for QW-QNN results

## Main result

**`ch5_walk_qnn/main_result.ipynb`** regenerates the main result of the thesis (Section 5.1.2): the continuous-time
walk QNN and the Liu–Arunachalam–Temme kernel, both after the discrete-logarithm step, against the classical
baselines (MLP, RBF-SVM, random forest). It writes **Table 5.1** (`results/tab_lat_main.tex`) and **Figure 5.1**
(`results/fig_lat_accuracy.pdf`), prints some tests quoted in the text, and compares every regenerated
accuracy with the value printed in the thesis. Running all takes about 5–10 minutes on a laptop CPU.

## Setup

```
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Python 3.11+. PennyLane/JAX are not needed: the benchmark suite's MLP and SVM are scikit-learn models with the
suite's settings, and are built from scikit-learn directly (`qwt/selection.py`).
