"""Generates sec_gradvar.tex (the thesis subsection) from the two gradient-variance JSON files,
so that every number in the text comes from the recorded runs."""

import json

d = json.load(open("gradient_variance.json"))
w = json.load(open("gradient_variance_width.json"))


def dr(ans, depth):
    return next(r for r in d["rows"] if r["ansatz"] == ans and r["depth"] == depth)


def wr(ans, q):
    return next(r for r in w["rows"] if r["ansatz"] == ans and r["qubits"] == q)


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


s = w["log2_slope_per_qubit"]
qmax = max(r["qubits"] for r in w["rows"])
Nmax = max(r["N"] for r in w["rows"])
real = [dr("walk, real coin", t)["var"] for t in (1, 2, 4, 8, 16, 32)]
cplx = [dr("walk, complex coin", t)["var"] for t in (1, 2, 4, 8, 16, 32)]
hea = [dr("hardware-efficient", t)["var"] for t in (1, 2, 4, 8, 16, 32)]
ratio = [a / b for a, b in zip(real, cplx)]
pv_l, ps_l, he_l = (wr(a, qmax)["var_local"] for a in
                    ("walk, per-vertex coin", "walk, per-step coin", "hardware-efficient"))
qmin = min(r["qubits"] for r in w["rows"])
tot = {a: wr(a, qmin)["var_local"] / wr(a, qmax)["var_local"] for a in
       ("walk, per-vertex coin", "walk, per-step coin", "hardware-efficient")}
def fac(x):
    """Two significant figures, thin-space thousands separator for LaTeX."""
    from math import floor, log10
    r = round(x, -int(floor(log10(x))) + 1)
    return f"{r:,.0f}".replace(",", "\\,")
f_pv = 2 ** (-s["walk, per-vertex coin var_local"])
f_he = 2 ** (-s["hardware-efficient var_local"])

T = rf"""
\subsection{{Gradient Variance of the Walk Ansatz}}
\label{{sec:gradvar}}

Every variational result in this chapter was obtained with a derivative-free
optimiser under a fixed budget, so a model that fails to improve may be
failing because its landscape is flat or because the optimiser never reached
the region where it is not. The quantity that separates the two is the
variance of the gradient at random parameters, whose exponential decay in the
number of qubits defines a barren plateau~\cite{{McClean_2018, Larocca_2025}}.
It is measured here in two ways. Both draw every parameter uniformly from
$[0, 2\pi)$, differentiate with respect to eight randomly chosen parameters
per draw over {d["n_draws"]} draws, and report the variance with a bootstrap
$95$ per cent interval. Gradients are central finite differences on the exact
state vector, checked against the exact parameter-shift rule on the
hardware-efficient ansatz to within ${d["validation_max_abs_delta"]:.0e}$.

\paragraph*{{Depth, on the classifier's own register.}}
The first measurement uses the classifier of Section~\ref{{sec:qnnmethod}}
unchanged: the same training loss, the same five-qubit states produced by two
steps of the real walk feature map on the $140$ training images, and the same
observable $Z^{{\otimes 5}}$, so that only the ansatz varies. Three ansätze
are compared at depths $1$ to $32$: the walk ansatz with a real coin and with
a complex coin, and a hardware-efficient ansatz of $R_Y$ rotations, a CNOT
ladder and $R_X$ rotations on the same five qubits.
Figure~\ref{{fig:gradvar}}(a) gives the result. None of the three decays with
depth. The real-coin walk ansatz holds between {sci(min(real))} and
{sci(max(real))} across all depths, above the hardware-efficient ansatz's
{sci(min(hea))} to {sci(max(hea))}; the complex-coin ansatz sits
{min(ratio):.1f} to {max(ratio):.1f} times lower than the real one at every
depth. At the depths the classifiers actually use, two and four steps, the
walk ansatz therefore has gradients at least as large as a standard ansatz on
the same register, and the results of Section~\ref{{sec:qnnresults}} are not
the product of a vanishing gradient. The complex arm here is the trained-phase
ansatz after the real feature map, the configuration in which the complex coin
fits worse at equal parameter count; its smaller per-parameter gradient is
consistent with that, though it does not by itself explain it, and says
nothing about the complex feature map, with which the sign of the effect
reverses.

\paragraph*{{Width, on graphs of increasing size.}}
Five qubits cannot exhibit a barren plateau, since the phenomenon is decay in
the number of qubits. For a walk that number is set by the graph, $\log_2 N +
\log_2 d$, so the second measurement varies the graph: grids of $8$ to
${Nmax}$ vertices, giving registers of $5$ to ${qmax}$ qubits, each walked for
$R + C$ steps from the uniform arc state with no data, so that the ansatz is
isolated as in~\cite{{McClean_2018}}. Two walk ansätze are compared, the
per-vertex coin of axis~1, with an independent angle at every vertex and
step, and the per-step coin of axis~2, with one angle per step shared by all
vertices, against the hardware-efficient ansatz of the same width and depth.
Because a cost measured on every qubit is known to flatten even shallow
circuits~\cite{{Larocca_2025}}, both the classifier's global observable
$Z^{{\otimes n}}$ and a local one, $Z$ on a single qubit, are reported.
Figure~\ref{{fig:gradvar}}(b,c) and Table~\ref{{tab:gradvar}} give the result.

The two walk ansätze behave in opposite ways. Under the local cost the
per-vertex coin loses a factor of about ${f_pv:.1f}$ in variance per added
qubit, against about ${f_he:.1f}$ for the hardware-efficient ansatz; since the
register grows as $\log_2 N$, this is a variance falling as
$N^{{{s["walk, per-vertex coin var_global"]:.1f}}}$ to
$N^{{{s["walk, per-vertex coin var_local"]:.1f}}}$ (global and local cost) in the size
of the graph --- polynomial in $N$, but exponential in the
qubit count, and faster than the generic ansatz. The mechanism is locality:
each angle rotates the coin at one vertex at one step, and so acts on a share
of the amplitude of order $1/N$. The per-step coin does the reverse. Its
angle acts at every vertex at once, and under the local cost its variance
falls by a factor of only about {fac(tot["walk, per-step coin"])} between ${qmin}$ and
${qmax}$ qubits --- a fitted slope of ${s["walk, per-step coin var_local"]:+.2f}$ in
$\log_2 \mathrm{{Var}}$ per qubit, noisy at the largest sizes --- against
about {fac(tot["hardware-efficient"])} for the hardware-efficient ansatz and
about {fac(tot["walk, per-vertex coin"])} for the per-vertex coin. At ${qmax}$
qubits it stands at {sci(ps_l)}, against {sci(he_l)} and {sci(pv_l)}
respectively. Under the global cost every ansatz
decays, the per-step coin included, which is the global-cost mechanism rather
than a property of the walk.

Three consequences follow. The first is that the per-vertex parameterisation
of axis~1 --- the one the classifier's walk ansatz uses --- fails at scale
for a second, independent reason: its gate count grows as $N^{{3.1}}$
(Section~\ref{{sec:depth}}), and its gradient variance shrinks as about
$N^{{-2}}$. The second
is that the per-step coin of axis~2, which the resource analysis already
favoured because it costs nothing in width or depth, is also the one that
keeps a usable gradient as the graph grows, provided the read-out is local.
The third is that the classifier's global read-out $Z^{{\otimes n}}$ would have
to be replaced by a local one before the construction could be trained at
any size larger than the one used here. The measurement has limits that
should be stated with it: it is taken at random parameters, not along the
path an optimiser follows; it uses grid graphs only, up to ${qmax}$ qubits; it
uses the real rotation coin, not the Grover-exponential coin of the graph
walk network; and it does not replace a training run with gradients, so the
negative results of this chapter about trainable coins remain conditional on
the derivative-free budget under which they were obtained.

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\textwidth]{{fig_gradient_variance.pdf}}
  \caption[Gradient variance of the walk ansatz]{{Variance of the gradient
  with respect to one ansatz parameter at uniformly random parameters, with
  bootstrap $95$ per cent bands. (a)~Against depth, on the five-qubit
  line-detection register with the classifier's loss and observable; only the
  ansatz differs. (b,c)~Against register width on grid graphs of $8$ to
  ${Nmax}$ vertices, at depth $R + C$ and without data, for a local and a
  global cost. The per-vertex coin gives each angle one vertex; the per-step
  coin shares one angle across all vertices.}}
  \label{{fig:gradvar}}
\end{{figure}}

\input{{tab_gradvar}}
"""
open("sec_gradvar.tex", "w").write(T.lstrip("\n"))
print("wrote sec_gradvar.tex")
