# implementation_5_hyperplanes — quantum-walk models on hyperplanes_diff (qml-benchmarks)

Data: exact regeneration of the suite's `hyperplanes_diff` (19 instances, k = 2..20 hyperplanes,
10 features from a 3-d latent space, 240 train / 60 test). Verified by reproducing the suite's
published SVC test accuracy on all 19 instances exactly (`data.py`).

Target (published, mean test accuracy over 19 instances): MLPClassifier 0.797,
DressedQuantumCircuit 0.786, DataReuploading 0.785, IQPKernel 0.776, ProjectedQuantumKernel 0.775, SVC 0.769.

## Models tried (in order; all hyperparameters chosen by 5-fold CV on the training set)

| model | file | CV (5 or 19 instances) | test (19 instances) |
|---|---|---|---|
| walk re-uploading network: CTQW on C_n, data-dependent walk times, trainable potentials, diagonal readout | `qwrn.py` | ~0.745 (5 inst.) | not run |
| multi-walker product network: H independent cycle walkers, product observable | `mwqw.py` | ~0.75-0.77 (5 inst.) | not run |
| star-graph walk energy product network (contains the exact parity-of-hyperplanes label) | `starwalk.py` | ~0.72-0.74 (5 inst.) | not run |
| **CTQW product fidelity kernel + SVM**, K = prod_i \|<0\|exp(-i s dx_i A_{C_n})\|0>\|^2, n in {8,16,64} | `walkkernel.py`, `final_kernel.py` | 0.808 | **0.795** |
| RBF-SVM control, identical protocol and C grid | `final_kernel.py` | 0.809 | 0.788 |
| walk kernel, grid extended to n in {3..8} (run 2, after seeing run-1 test scores) | `final_kernel2.py` | 0.809 | 0.775 |

Variational walk models overfit relative to the MLP (e.g. k = 20: train 0.88-0.93, validation 0.61-0.64,
MLP train 0.955 / validation 0.767). The walk kernel at short walk times approximates an RBF kernel
(|G_n(t)|^2 ~ exp(-2t^2)); CV preferred the small cycle n = 8, whose revivals make it differ from RBF.

## Outcome
Best result 0.795 vs published MLP 0.797: a statistical tie, not a win (one test point = 1.7 %).
Beats every published quantum model on this family (best: 0.786) and the RBF control (0.788)
under the same protocol, but the RBF control shows that part of the margin over the published
SVC (0.769) comes from a finer C grid, not from the walk. Run 2 shows CV-based selection noise
of the same order as all the gaps in the table. No further tuning was done against the test sets.

## Thesis outputs (`thesis_outputs.py`, 25 Sep 2026)
Re-runs the run-1 protocol of `final_kernel.py` with per-test-point predictions kept, asserts the numbers equal
`kernel_results.csv`, re-checks the data (published SVC test accuracy reproduced on 19/19 instances), and compares
with the suite's published per-instance results (`QMLB_REPO`, default `./qml-benchmarks`).
Writes `tab_kernel_hp.tex`, `fig_kernel_hp.{pdf,png}` (copied to `thesis/`) and `kernel_hp_stats.json`.

- **Paired tests over the 19 instances (Wilcoxon):**
  - vs MLP: 9/10 wins/losses, p = 0.47.
  - vs dressed circuit: p = 0.55.
  - vs IQP and projected kernels: p = 0.13 each.
  - vs the suite's SVC: p = 0.005.
  - vs the RBF control run under this protocol: pooled McNemar 29:21, p = 0.32.
- **The walk product kernel is numerically an RBF kernel:** |G_n(t)|^2 = 1 - 2t^2 + O(t^4), so K ~ exp(-2 s^2 |x-x'|^2).
  At the selected (s, n), the off-diagonal Gram entries differ from RBF(gamma = 2 s^2) by at most 0.055, with correlation >= 0.999.
  CV picked n = 8 on all 19 instances. The revivals of C_8 do not matter at the selected times.
  This supersedes the "differ from RBF" remark above.
