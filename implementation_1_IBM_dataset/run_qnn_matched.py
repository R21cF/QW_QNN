"""
Real versus complex coins at matched depth AND matched parameter count.

In the main grid (run_qnn.py) a complex coin trains its phase as a second
parameter per vertex, so a complex ansatz carries twice the parameters of a
real one at the same number of walk steps; parameter count and depth cannot
both be matched. Here the complex arm freezes the phase ramp at a value fixed
in advance and trains only the rotation angle, so both arms carry exactly one
parameter per vertex per step and differ only in whether the coin is real.

Design: feature map {real, complex} x ansatz depth {2, 4} x coin {real,
complex with fixed phase}. The pre-registered phase is pi/2 (complex at both
vertex degrees on the 2x4 grid; pi would make the coin real). Three further
phases are run as a robustness check and are reported, not selected from.

Both arms have the same number of weights, so seed s gives identical initial
angles in both, and the comparison is paired seed by seed.

Writes qnn_matched.json. Needs numpy and scipy only.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

from ibm_dataset import load
from run_qnn import WalkRunner, evaluate

SEEDS = 10
PHASE_MAIN = np.pi / 2
PHASE_CHECK = {"pi/4": np.pi / 4, "2pi/3": 2 * np.pi / 3, "3pi/4": 3 * np.pi / 4}


def model(fm_cplx, steps, phase):
    """phase None -> real ansatz coin; otherwise complex with that fixed phase."""
    return WalkRunner(2, steps, fm_cplx, phase is not None, "walk",
                      fixed_phase=phase)


def label(fm_cplx, steps, phase_name):
    fm = "cplx" if fm_cplx else "real"
    coin = "real" if phase_name is None else f"cplx, phi={phase_name}"
    return f"walk FM ({fm}) + walk ansatz ({coin}, {steps} steps)"


def verify(Xtr):
    """Batched path against the per-sample forward, for the new models."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for fm in (False, True):
        for steps in (2, 4):
            r = model(fm, steps, PHASE_MAIN)
            w = rng.uniform(0, 2 * np.pi, r.n_weights)
            b = r.outputs(r.encode(Xtr[:6]), w)
            s = np.array([r.m.forward(x, w) for x in Xtr[:6]])
            worst = max(worst, float(np.abs(b - s).max()))
    return worst


def paired(a, b):
    """a, b: per-seed test accuracies of two arms with identical initial weights."""
    d = np.asarray(b) - np.asarray(a)
    wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    n = wins + losses
    sign_p = binomtest(wins, n, 0.5).pvalue if n else 1.0
    try:
        wil_p = float(wilcoxon(a, b, zero_method="wilcox").pvalue) if n else 1.0
    except ValueError:
        wil_p = 1.0
    return dict(mean_diff=float(d.mean()), std_diff=float(d.std()),
                wins=wins, losses=losses, ties=len(d) - n,
                sign_test_p=float(sign_p), wilcoxon_p=wil_p)


def per_seed(r):
    """Recompute per-seed test accuracy from a result row is not stored; keep it."""
    return r["test_accs"]


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    w = verify(Xtr)
    print(f"batched vs per-sample, fixed-phase models: max |delta| = {w:.2e}")
    if w > 1e-11:
        sys.exit("verification failed")

    # evaluate() does not keep per-seed accuracies; wrap it to recover them
    import run_qnn

    def run(name, runner):
        accs = []
        orig = run_qnn.train_once

        def spy(*a, **k):
            out = orig(*a, **k)
            accs.append(out[1])
            return out

        run_qnn.train_once = spy
        try:
            row = evaluate(name, runner, Xtr, ytr, Xte, yte, SEEDS)
        finally:
            run_qnn.train_once = orig
        row["test_accs"] = accs
        row.pop("histories", None)
        print(f"  {name:<58} p={row['weights']:<3} "
              f"test {row['test_acc_mean']*100:5.1f} +/- {row['test_acc_std']*100:4.1f}"
              f"   [{row['seconds']}s]")
        return row

    rows, comparisons = [], []
    print(f"\n{len(ytr)} train / {len(yte)} test, {SEEDS} seeds, COBYLA maxiter "
          f"{run_qnn.MAXITER}\n")
    for fm in (False, True):
        for steps in (2, 4):
            real = run(label(fm, steps, None), model(fm, steps, None))
            cplx = run(label(fm, steps, "pi/2"), model(fm, steps, PHASE_MAIN))
            real["arm"], cplx["arm"] = "real", "cplx"
            for r in (real, cplx):
                r.update(fm="cplx" if fm else "real", steps=steps)
            cplx["phase"] = "pi/2"
            rows += [real, cplx]
            c = paired(real["test_accs"], cplx["test_accs"])
            c.update(fm="cplx" if fm else "real", steps=steps, phase="pi/2",
                     params=real["weights"])
            comparisons.append(c)
            for pname, ph in PHASE_CHECK.items():
                chk = run(label(fm, steps, pname), model(fm, steps, ph))
                chk.update(arm="cplx", fm="cplx" if fm else "real", steps=steps,
                           phase=pname)
                rows.append(chk)
                c = paired(real["test_accs"], chk["test_accs"])
                c.update(fm="cplx" if fm else "real", steps=steps, phase=pname,
                         params=real["weights"])
                comparisons.append(c)

    # the real-coin rows must reproduce the main grid exactly
    main = {r["name"]: r for r in json.load(open("qnn_results.json"))}
    repro = {}
    for r in rows:
        if r["arm"] != "real":
            continue
        key = ("walk FM (%s) + walk ansatz (real%s)"
               % (r["fm"], ", 4 steps" if r["steps"] == 4 else ""))
        if key in main:
            repro[key] = abs(main[key]["test_acc_mean"] - r["test_acc_mean"])
    print("\nreproduction of the main grid's real-coin rows:",
          {k: f"{v:.2e}" for k, v in repro.items()})

    json.dump(dict(rows=rows, comparisons=comparisons, reproduction=repro,
                   seeds=SEEDS, maxiter=run_qnn.MAXITER,
                   phase_main="pi/2", test_size=len(yte)),
              open("qnn_matched.json", "w"), indent=2)
    print("\npaired comparisons (complex minus real, test accuracy):")
    for c in comparisons:
        print(f"  FM {c['fm']:4s} {c['steps']} steps p={c['params']:<3} phi={c['phase']:<5} "
              f"diff {c['mean_diff']*100:+5.1f} +/- {c['std_diff']*100:4.1f}   "
              f"W/L/T {c['wins']}/{c['losses']}/{c['ties']}   "
              f"sign p={c['sign_test_p']:.3f}  wilcoxon p={c['wilcoxon_p']:.3f}")
    print("\nwrote qnn_matched.json")
