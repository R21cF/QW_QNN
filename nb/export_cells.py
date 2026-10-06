"""Export the shared code cells of implementation_7_qwqnn_lat/qwqnn_lat_colab.ipynb to plain files.

The notebook is the single source for the LAT data generator (c2_data.py), the walk/kernel models
(c3_models.py) and the model-selection suite (c4_suite.py). The confirmatory runs in
implementation_7_qwqnn_lat and the walk-DLP pipeline in implementation_8_walk_dlp `exec` these
files. Run this after editing one of those cells; `--check` fails if the files are out of date.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "..", "implementation_7_qwqnn_lat", "qwqnn_lat_colab.ipynb")
CELLS = {                                 # first line of the cell  ->  exported file
    "# ---- data:": "c2_data.py",
    "# ---- models": "c3_models.py",
    "# ---- qml-benchmarks models": "c4_suite.py",
}


def exported():
    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    out = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for prefix, name in CELLS.items():
            if src.startswith(prefix):
                out[name] = (f"# Exported from qwqnn_lat_colab.ipynb by nb/export_cells.py; do not edit here.\n"
                             + src.rstrip("\n") + "\n")
    missing = set(CELLS.values()) - set(out)
    if missing:
        sys.exit(f"cells not found in notebook: {sorted(missing)}")
    return out


if __name__ == "__main__":
    files = exported()
    if "--check" in sys.argv:
        stale = [n for n, s in files.items()
                 if not os.path.exists(os.path.join(HERE, n)) or open(os.path.join(HERE, n), encoding="utf-8").read() != s]
        if stale:
            sys.exit(f"stale: {stale} -- run python nb/export_cells.py")
        print("nb/ is up to date with the notebook")
    else:
        for name, src in files.items():
            with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(src)
            print("wrote", name)
