# nb — shared code exported from the LAT notebook

`c2_data.py`, `c3_models.py` and `c4_suite.py` are the data, models and model-selection cells of
`../implementation_7_qwqnn_lat/qwqnn_lat_colab.ipynb`, written out as plain files by `export_cells.py`.
They are `exec`'d by

- `implementation_7_qwqnn_lat/confirm_run.py`, `confirm_run2.py` (pre-registered confirmatory runs), and
- `implementation_8_walk_dlp/scan_dt_ct.py`, `pipeline_walk_dlp.py` (walk-DLP scan and pipeline).

The notebook is the single source. Edit the cell there, then run

    python nb/export_cells.py           # rewrite the three files
    python nb/export_cells.py --check   # exit 1 if they differ from the notebook

`c4_suite.py` imports `qml_benchmarks` only if it is installed; without it the suite's own models
(`iqp_kernel`, `projected_kernel`, `data_reuploading`, `mlp`, `svc_rbf`) are unavailable, while the
walk QNN, the LAT kernel and the `*_full` selectors used by the pipeline need only numpy, scipy and
scikit-learn.

Check: with these files, `ct_walk_qnn_full` and `lat_kernel_full` on the exact exponents reproduce the
per-test-point predictions stored in `implementation_7_qwqnn_lat/lat_results.json` for all 20
development draws (n = 8, 10, 12, 16; draws 0–4). This is the K = ∞ identity check of `pipeline_walk_dlp.py`.
