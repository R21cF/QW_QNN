# implementation_6_interacting_walkers — interacting multi-walker QW network on hyperplanes_diff

`imw_hyperplanes_colab.ipynb` is self-contained: it clones qml-benchmarks, regenerates the 19
hyperplanes_diff instances (and checks them against the suite's published SVC scores, 19/19),
defines the model, runs the pre-registered protocol for two arms, and compares with the
published leaderboard.

Model: 3 distinguishable walkers on the cycle C8 (9 qubits); per layer, independent CTQWs for
learned input-dependent times, then an on-site interaction exp(-i g sum_{a<b} delta(pos_a,pos_b))
that entangles the walkers (cross-feature entanglement). Readout: trainable product observable
plus single-walker terms. Control arm: identical model with the interaction off (product state).

Pre-registered (24 Sep 2026): lr 0.01, 1500 steps, grid layers {2,4} x weight decay {1e-2,1e-1};
5-fold CV on training data; each test set scored once; 5 seeds.

## Run on Colab
Runtime -> Change runtime type -> GPU; Runtime -> Run all; allow Google Drive. Results go to
MyDrive/imw_hyperplanes/imw_results.json after every instance; re-running resumes.

## Run locally (in a venv)
    python -m venv .venv && .venv\Scripts\activate      (Windows)
    pip install jax optax numpy scipy pandas scikit-learn jupyter
    jupyter nbconvert --to notebook --execute imw_hyperplanes_colab.ipynb   (set USE_DRIVE = False first)

`imw.py` is the same model as a standalone module (float64 version used for development).
