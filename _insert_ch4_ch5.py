# -*- coding: utf-8 -*-
"""
Splice the QNN methodology and results into draft.tex.

Each block replaces a bare heading (or a heading plus its planning comment)
with the heading followed by prose.  Every replacement is asserted to match
exactly once, so a silent mis-splice is impossible.
"""

import io, os, sys

P = os.path.expanduser("~/mnt/thesis/draft.tex")
s = io.open(P, encoding="utf-8").read()
orig = s
done = []


def rep(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        print(f"FAIL [{tag}]: {n} occurrences"); sys.exit(1)
    s = s.replace(old, new)
    done.append(tag)


# ======================================================================
# 4.2  Simulation environment
# ======================================================================

rep(
r"""\section{Simulation Environment and Tooling}
% instructions.md: Qiskit preferred; another framework is acceptable where it
% is the better fit, provided the algorithms work.
\subsection{Qiskit: Circuit Construction and Simulation}
\subsection{Statevector Simulation and Its Limits}
\subsection{Reproducibility: Seeds, Datasets and Parameter Records}""",
r"""\section{Simulation Environment and Tooling}
% instructions.md: Qiskit preferred; another framework is acceptable where it
% is the better fit, provided the algorithms work.

\subsection{Qiskit: Circuit Construction and Simulation}

All circuits in this thesis are built with Qiskit~2.5. A walk on a graph of
$N$ vertices and maximum degree $d$ occupies two registers: a position
register of $\lceil \log_2 N \rceil$ qubits and a coin register of
$\lceil \log_2 d \rceil$ qubits, with the coin register taking the low-order
positions so that the basis state $\ket{v}\ket{c}$ carries index
$v\,d_{\mathrm{pad}} + c$ under Qiskit's little-endian convention.

A single walk step is assembled as the product $U = S\,(I \otimes C)$ and
appended to the circuit as one dense \texttt{UnitaryGate}. This is the general
construction: it makes no assumption about the graph, and it is correct for
the irregular graphs the applications use. It is also, as
Section~\ref{sec:depth} reports, extremely expensive once transpiled, and the
depth figures given there should be read as upper bounds rather than as the
cost of a walk step as such.

Training a variational model calls the forward pass tens of thousands of
times, which Qiskit's \texttt{Statevector} is too slow to support. The
optimisation loops therefore run on a small \textsc{numpy} statevector core
written for this work, and a cross-check suite asserts that this core
reproduces Qiskit's own result: the encoding circuit, the full gate model's
statevector and its expectation value, and the walk model against an
equivalent circuit of dense step unitaries all agree to within $10^{-16}$.
The speed costs nothing in fidelity.

\subsection{Statevector Simulation and Its Limits}

Every number reported in Chapter~\ref{ch:results} comes from noiseless
statevector simulation. Nothing is sampled, so there is no shot noise, and
nothing is subjected to a device noise model. Two consequences follow and
both matter for how the results should be read.

First, the accuracies reported are a ceiling. A real device would add
readout error and gate infidelity, and the expectation values that drive the
classifier would additionally carry sampling error of order $1/\sqrt{M}$ in
the shot count $M$. A model that fails in statevector simulation will not be
rescued by hardware; a model that succeeds there has only cleared the
easiest bar.

Second, the tractable problem size is set by the dimension $N d$ rather than
by the qubit count alone. The walk models in this chapter occupy five qubits
and a Hilbert space of dimension~32, which is trivial to simulate; the graph
classification benchmarks of Section~\ref{sec:kernelmethod} reach a few
hundred dimensions and remain comfortable. What statevector simulation cannot
reach is the regime in which a quantum advantage could even be argued for,
and no claim of advantage is made anywhere in this thesis on the strength of
these simulations.

\subsection{Reproducibility: Seeds, Datasets and Parameter Records}

Every stochastic element is seeded and the seeds are stated. Dataset
generation uses seed~42 and the train/test partition seed~246, both inherited
from the reference implementation so that the split is the one the baseline
was measured on. Weight initialisations use seeds $1000$ to $1009$, giving
the ten independent restarts over which all reported accuracies are averaged;
reporting a single run would be indistinguishable from reporting the best
one. Transpilation is seeded at~7 so that the resource figures are stable
across invocations.

The simulation code, the validation suites and the scripts that generate
every figure and table in this chapter are kept alongside the manuscript
source. Results are written to a machine-readable record rather than
transcribed by hand, and the tables in Chapter~\ref{ch:results} are generated
from that record.""",
    "4.2 tooling")


# ======================================================================
# 4.4  Application I methodology
# ======================================================================

rep(
r"""\section{Application I---Quantum Neural Networks and Variational Classifiers}
\subsection{Analysis of the Existing Walk-Based Architecture}
\subsection{Walk-Based Feature Maps}
\subsection{Substitution of the Modified Walk}
\subsection{The Learned Coin and the Limits of the Coin Modification}
\subsection{Datasets, Baselines and Evaluation Protocol}""",
r"""\section{Application I---Quantum Neural Networks and Variational Classifiers}
\label{sec:qnnmethod}

\subsection{Analysis of the Existing Walk-Based Architecture}

The walk-based neural network of Dernbach et al.~\cite{Dernbach_2018,
Dernbach_2019} is the reference architecture for this application. Its
premise is that the diffusion operator of a graph neural network need not be
fixed by the graph's geometry alone, as it is in a diffusion-convolutional or
graph-convolutional network, but can be learned. A discrete-time quantum walk
supplies exactly such a tunable diffusion: the coin operator decides how
amplitude at a vertex is distributed over the incident edges, so learning the
coin learns the diffusion.

In the later of the two formulations, a \emph{bank} produces a coin for every
vertex at every step as a function of the node features, so the walk varies
both spatially across the graph and temporally along its own trajectory, and
two structurally identical graphs carrying different features induce
different walks. The resulting position distributions are assembled into a
diffusion matrix, that matrix diffuses the feature matrix, and the diffused
features pass to the remaining layers. Gradients reach the coins by
backpropagation through the walk.

Two features of that construction bear directly on what this thesis can add.
The first is that the coins are restricted to real elementary reflections of
the form $I - 2ww^{\mathsf{T}}/(w^{\mathsf{T}}w)$. The authors are explicit
that this is a computational convenience: a general coin would have to be
projected back onto the unitary group after every update, which costs a
singular value decomposition per vertex per step, whereas an elementary
reflection is unitary by construction and differentiates cheaply. The
restriction is therefore not a claim that real coins suffice, and the
question of what complex coins would add is left open. That question is
axis~2 of this thesis.

The second is that the architecture is not a quantum circuit. The walk is
simulated classically as a tensor contraction inside a classical neural
network, and the quantum walk enters as a source of a differentiable
diffusion operator rather than as a computation performed on a quantum
device. Nothing in the construction requires a quantum computer, and nothing
about it speaks to circuit depth, qubit count or hardware feasibility. The
present work therefore does not reproduce that architecture; it builds a
circuit-native counterpart, in which the walk is a unitary acting on a
register and the model is a variational quantum classifier in the standard
sense. The comparison to be drawn against Dernbach et al.\ is one of
construction rather than of measured performance, and the numbers in
Chapter~\ref{ch:results} are not commensurable with theirs.

\subsection{Walk-Based Feature Maps}

A feature map encodes a classical vector $x$ as a state
$\ket{\Phi(x)} = U_\Phi(x)\ket{\psi_0}$. The conventional choices are product
encodings: the $Z$ feature map applies a layer of Hadamards followed by a
phase rotation $P(2x_i)$ on qubit $i$, repeated, and contains no entangling
gate at all. Whatever relational structure the data carries must therefore be
supplied later, by the ansatz.

The walk-based feature map proposed here instead lets the data drive the coin
of a walk running on a graph that expresses the data's own structure,
\begin{equation}
  U_\Phi(x) \;=\; \prod_{t=1}^{T_\Phi} S \, \bigl(I \otimes C_t(x)\bigr),
  \qquad
  \bigl[C_t(x)\bigr]_v \;=\; K\bigl(\deg v,\; \alpha\, x_v\bigr),
  \label{eq:walkfm}
\end{equation}
where $[\,\cdot\,]_v$ denotes the block acting at vertex $v$ and
$K(\delta,\theta)$ is the one-parameter coin of
Section~\ref{sec:coindesign}, a chain of Givens rotations through a common
angle $\theta$ on the $\delta$-dimensional subspace of real incident
directions. The walker begins in the uniform superposition over all real
(vertex, direction) pairs; localising it at a single vertex would make the
encoding depend on an arbitrary choice of origin, which an image or a
molecule does not possess.

The structural claim for~\eqref{eq:walkfm} is that adjacency enters through
the shift rather than through a design decision. A product encoding is blind
to which features are neighbours, and the ansatz has to be told; the walk
cannot be blind to it, because the shift operator is the adjacency. Whether
that structural prior is worth its cost is the question
Section~\ref{sec:qnnresults} answers.

\subsection{Substitution of the Modified Walk}

Both modification axes enter the model at two independent points, and the
experimental design keeps them separable. The feature map may use a real or a
complex coin, and the ansatz may use a real or a complex coin, giving four
combinations, each run against the same baselines on the same partition.

The ansatz is a second walk on the same graph whose coin angles are free
parameters rather than data,
\begin{equation}
  W(\theta) \;=\; \prod_{t=1}^{T_W} S \, \bigl(I \otimes C_t(\theta)\bigr),
  \qquad
  \bigl[C_t(\theta)\bigr]_v \;=\; K\bigl(\deg v,\; \theta_{t,v}\bigr),
\end{equation}
so that a walk of $T_W$ steps on $N$ vertices carries $T_W N$ real parameters,
or $2 T_W N$ when the coins are complex. The classifier output is the
expectation of the all-$Z$ observable in the final state,
$f_\theta(x) = \bra{\Phi(x)} W^\dagger(\theta)\, Z^{\otimes n}\, W(\theta)
\ket{\Phi(x)}$, and the predicted label is $\operatorname{sgn} f_\theta(x)$.
A rotation ansatz of the conventional kind, acting on the walk register
rather than through the walk, is also run, so that the feature map's
contribution can be separated from the ansatz's.

\subsection{The Learned Coin and the Limits of the Coin Modification}
\label{sec:coinlimits}

Extending the coin from $O(\delta)$ to $U(\delta)$ appears to buy a great
deal: the real orthogonal group has $\delta(\delta-1)/2$ parameters against
the unitary group's $\delta^2$. Most of that apparent gain is not available,
and the reason has to be stated before any complex-coin result is reported.

A coin block multiplied by $e^{i\alpha}I$ is the same coin. A phase vector
whose entries are all equal is therefore a global phase on that block and
changes no measurable quantity whatsoever; only phase \emph{differences}
within a block have any effect. Any construction that generates complex coins
by attaching a single common phase is consequently a null modification
dressed as a contribution, and the parameterisation used here attaches
$\operatorname{diag}(e^{i\varphi k})_{k=0}^{\delta-1}$ precisely so that the
phases cannot collapse to a constant. The validation suite asserts the
degenerate case directly: a uniform coin phase leaves the position
distribution unchanged to $4 \times 10^{-16}$.

A second consequence is methodological. A complex coin carries two real
numbers per vertex where a real coin carries one, so a complex model
outscoring a real model at the same number of \emph{steps} has also been
given twice the parameters, and the comparison establishes nothing. Every
complex condition reported in Chapter~\ref{ch:results} is therefore matched
against a real condition of equal parameter count, obtained by doubling the
number of walk steps. This control turns out to be decisive rather than
decorative.

\subsection{Datasets, Baselines and Evaluation Protocol}
\label{sec:qnnprotocol}

The task is the line-detection problem used in IBM's quantum machine learning
course~\cite{IBMQML2026}, chosen so that the walk models are measured against
a published construction on its own ground rather than on one selected to
favour them. Each sample is a $2 \times 4$ pixel grid flattened row-major.
Two pixels are lit, at value $\pi/2$; every other pixel is drawn uniformly
from $(0, \pi/4)$. The lit pair is either horizontally adjacent within a row,
labelled $-1$, or vertically adjacent within a column, labelled $+1$. Two
hundred images are generated and split 140/60 into training and test sets.
Pixel values are used directly as rotation angles. The walk models run on the
grid graph itself: eight vertices, degrees two and three, maximum degree
three, giving a padded coin dimension of four and a five-qubit register.

Three sets of baselines are used. The first two are the course's own models,
both on eight qubits with a $Z$ feature map and a sixteen-parameter ansatz of
$R_Y$ rotations, a CNOT layer and $R_X$ rotations. They differ only in the
CNOT layer: the first entangles qubits $(0,1)$, $(1,2)$ and $(2,3)$, and the
second additionally entangles $(4,5)$, $(5,6)$ and $(6,7)$, which are exactly
the within-row adjacencies of the grid. Reproducing the reported gap between
them is the check that the present implementation matches the reference.

The third set is classical and is the one the quantum literature most often
omits. A majority-class predictor, logistic regression, and support vector
machines with linear and radial-basis-function kernels are fitted to the raw
eight pixels on the same partition. Without them there is no way to tell
whether a quantum model scoring highly has done anything difficult.

Training minimises the mean squared error between $f_\theta(x)$ and the label
over the full training set, using COBYLA with a budget of 500 objective
evaluations. Each model is trained from ten independent random weight
initialisations drawn uniformly from $[0, 2\pi]$, and results are reported as
the mean and standard deviation of test accuracy over those ten runs.""",
    "4.4 QNN methodology")


io.open(P, "w", encoding="utf-8").write(s)
print("applied:", ", ".join(done))
print(f"bytes {len(orig)} -> {len(s)}")
