"""tab_qnn_matched.tex from qnn_matched.json (no plotting libraries needed)."""

import json

d = json.load(open("qnn_matched.json"))
rows = d["rows"]


def find(fm, steps, arm, phase=None):
    for r in rows:
        if r["fm"] == fm and r["steps"] == steps and r["arm"] == arm and \
                (arm == "real" or r.get("phase") == phase):
            return r
    raise KeyError((fm, steps, arm, phase))


def comp(fm, steps, phase):
    for c in d["comparisons"]:
        if c["fm"] == fm and c["steps"] == steps and c["phase"] == phase:
            return c
    raise KeyError((fm, steps, phase))


def fmt(v):
    v = round(v, 1)
    return "0.0" if v == 0 else f"{v:+.1f}"


def pm(r):
    return f"${r['test_acc_mean']*100:.1f} \\pm {r['test_acc_std']*100:.1f}$"


lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption[Real against complex coins at matched depth and parameter count]"
    r"{Real against complex ansatz coins on the line-detection task, with depth "
    r"and parameter count matched: the complex coin has its phase ramp frozen "
    r"and trains only its rotation angle, so both arms carry one parameter per "
    r"vertex per step. Test accuracy (\%) over " + str(d["seeds"]) + r" weight "
    r"initialisations; the same seed gives both arms identical initial angles, so "
    r"$\Delta$ is the paired difference (complex minus real) and W/L counts the "
    r"seeds on which the complex arm wins or loses. The pre-registered phase is "
    r"$\varphi = \pi/2$; the last column is the range of $\Delta$ over three further "
    r"phases ($\pi/4$, $2\pi/3$, $3\pi/4$), reported as a robustness check and not "
    r"selected from.}",
    r"\label{tab:qnnmatched}",
    r"\small",
    r"\begin{tabular}{llrcccc}",
    r"\hline",
    r"Feat.\ map & Steps & Params & Real coin & Complex, $\pi/2$ & "
    r"$\Delta$ (W/L) & $\Delta$, other \\",
    r"\hline",
]
for fm in ("real", "cplx"):
    for steps in (2, 4):
        re_, cx = find(fm, steps, "real"), find(fm, steps, "cplx", "pi/2")
        c = comp(fm, steps, "pi/2")
        others = [comp(fm, steps, p)["mean_diff"] * 100
                  for p in ("pi/4", "2pi/3", "3pi/4")]
        lines.append(
            f"{fm} & {steps} & {re_['weights']} & {pm(re_)} & {pm(cx)} & "
            f"${fmt(c['mean_diff']*100)}$ ({c['wins']}/{c['losses']}) & "
            f"${fmt(min(others))}$ to ${fmt(max(others))}$ \\\\")
lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
open("tab_qnn_matched.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
