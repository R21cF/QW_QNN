# -*- coding: utf-8 -*-
"""Splice the Chapter 5 QNN results, the cross-reference labels and the three
missing bibliography entries."""

import io, os, sys

TEX = os.path.expanduser("~/mnt/thesis/draft.tex")
BIB = os.path.expanduser("~/mnt/thesis/draft_references.bib")
s = io.open(TEX, encoding="utf-8").read()
orig_len = len(s)
done = []


def rep(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        print(f"FAIL [{tag}]: {n} occurrences"); sys.exit(1)
    s = s.replace(old, new)
    done.append(tag)


# ------------------------------------------------- cross-reference labels --

rep(r"\chapter{RESULTS AND DISCUSSION}",
    "\\chapter{RESULTS AND DISCUSSION}\n\\label{ch:results}", "label results")

rep(r"""\subsection{The Coin Operator as a Design Space}""",
    "\\subsection{The Coin Operator as a Design Space}\n\\label{sec:coindesign}",
    "label coindesign")

rep(r"""\section{Application II---Quantum Kernel Methods}""",
    "\\section{Application II---Quantum Kernel Methods}\n\\label{sec:kernelmethod}",
    "label kernelmethod")


# --------------------------------------------------------------- 5.1.3 ----

rep(
r"""\subsection{Circuit Depth and Qubit Count}
% The measured overhead of the modified walk against the generic step costed
% in Sec.~2.4.3.""",
r"""\subsection{Circuit Depth and Qubit Count}
\label{sec:depth}
% The measured overhead of the modified walk against the generic step costed
% in Sec.~2.4.3.

The walk-based classifier of Section~\ref{sec:qnnmethod} is more economical
than its baseline in qubits and enormously more expensive in depth, and the
second fact dominates the first.

On the eight-vertex grid graph the walk occupies five qubits, three for
position and two for the coin, against the eight the product encoding needs
for the same eight features. Transpiled to the basis
$\{\mathrm{CX}, R_Z, \sqrt{X}, X\}$ at optimisation level~3, however, a
feature map of two steps followed by an ansatz of two steps reaches a depth
of $6119$ with $1691$ two-qubit gates, where the baseline circuit reaches a
depth of $21$ with $6$. The complex-coin variant is indistinguishable at
$6104$ and $1685$: the phases are free, and it is the walk that is expensive.

Two conclusions follow, and they should not be conflated. The first concerns
this construction and is damning: three qubits saved against roughly three
hundred times the depth is not a trade any device would accept, and against
the IBM course's own guidance that a transpiled two-qubit depth of fifty to
sixty is state of the art, these circuits are some two orders of magnitude
beyond what present hardware can execute. Whatever the classification results
say, the walk-based classifier as built here is a simulation-only object.

The second concerns walks in general and is not settled by the figure above.
The depth reported is the cost of \emph{synthesising} each step as a dense
$32 \times 32$ unitary, which is what the general construction of
Section~\ref{sec:qiskittooling} does for an arbitrary graph. It is an upper
bound on the cost of a walk step, not a measurement of it. Efficient circuit
constructions for discrete-time walks are known --- Douglas and
Wang~\cite{Douglas_2007} for graphs admitting a suitable colouring, and the
broader treatment in Loke's thesis~\cite{Han_2017} --- and they do not
synthesise a dense operator. Whether the grid graph used here admits such a
construction, and what depth it would actually reach, is taken up in
Section~\ref{sec:efficientcircuit}; the figures in this section stand as the
naive baseline that work has to beat.""",
    "5.1.3 depth")

rep(r"""\subsection{Qiskit: Circuit Construction and Simulation}""",
    "\\subsection{Qiskit: Circuit Construction and Simulation}\n"
    "\\label{sec:qiskittooling}", "label qiskittooling")


# ----------------------------------------------------------------- 5.2 ----

rep(
r"""\section{Quantum Neural Networks and Variational Classifiers}
\subsection{Training Behaviour}
\subsection{Classification Results}""",
r"""\section{Quantum Neural Networks and Variational Classifiers}
\label{sec:qnnresults}

\subsection{Training Behaviour}

Figure~\ref{fig:qnntraining} shows the training objective against COBYLA
evaluation count for the four models that carry the argument, as a median
over ten weight initialisations with the interquartile range shaded.

All four descend smoothly and none shows the flat landscape characteristic of
a barren plateau, which is unsurprising at five to eight qubits and sixteen
to thirty-two parameters; nothing here probes the regime where trainability
becomes the binding constraint. The differences are in where they stop. The
baseline converges fastest, reaching its floor near evaluation~130. The
real-coin walk model at thirty-two parameters descends more slowly but
continues to improve until roughly evaluation~250 and settles only slightly
above the baseline. The real-coin walk model at sixteen parameters stalls
early, at a visibly higher objective, which is a capacity limit rather than
an optimisation failure.

The complex-coin model is the informative case. At thirty-two parameters ---
the same budget as the real-coin model it should be compared with --- it
converges to a \emph{higher} training error, and the gap opens early and
never closes. Its disadvantage is therefore not a generalisation effect that
a test set happens to expose; the complex model fits the training data less
well than the real model of equal size. Section~\ref{sec:coinlimits} predicted
that some of the apparent freedom in $U(\delta)$ is unavailable, and this is
what that looks like during optimisation: the additional parameters enlarge
the search space without enlarging the set of reachable behaviours enough to
pay for it.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_qnn_training.pdf}
  \caption[Training behaviour of the walk-based classifiers]{Training mean
  squared error against COBYLA objective evaluation on the line-detection
  task. Curves are medians over ten weight initialisations; bands give the
  interquartile range. The complex-coin model converges to a higher training
  error than the real-coin model of equal parameter count.}
  \label{fig:qnntraining}
\end{figure}

\subsection{Classification Results}

Table~\ref{tab:qnn} and Figure~\ref{fig:qnnaccuracy} give test accuracy for
every model, as a mean and standard deviation over ten weight
initialisations.

\paragraph{The implementation reproduces the reference.}
The two baseline models behave as the course reports. Its deliberately
inadequate ansatz, entangling only three of the six within-row pixel pairs,
reaches $71.3 \pm 11.2$ per cent; the full ansatz reaches
$99.0 \pm 3.0$ per cent with a best run at $100$. The course quotes
$60$ and $100$ per cent respectively. The discrepancy on the weaker model is
a budget effect and not a disagreement: the present protocol allows 500
objective evaluations against the course's 100, and the additional budget
lets the handicapped model recover part of its deficit. The full model
matches exactly, and the gap between the two --- the course's pedagogical
point, that the entangling pattern must match the grid's adjacency ---
reproduces cleanly.

\paragraph{The walk-based feature map matches the baseline on fewer qubits.}
A real-coin walk feature map followed by a real-coin walk ansatz of four
steps reaches $99.0 \pm 2.0$ per cent on \emph{five} qubits, statistically
indistinguishable from the eight-qubit baseline's $99.0 \pm 3.0$. This is the
chapter's positive result, and its interest lies less in the qubit count than
in how the two models come by their structure. The baseline attains its
accuracy only after its CNOT layer has been hand-matched to the horizontal
adjacencies of the pixel grid; with the wrong entangling pattern it collapses
to $71$ per cent. The walk model is never told which pixels are neighbours,
because the shift operator is the adjacency. The structural prior that the
course installs by design is, in the walk formulation, a property of the
operator.

\paragraph{Complex coins do not help.}
Every comparison that holds parameter count fixed favours the real coin, and
several do so by a wide margin. At thirty-two parameters with a real feature
map, the real-coin ansatz reaches $99.0 \pm 2.0$ against the complex-coin
ansatz's $92.7 \pm 4.4$. At sixteen parameters, a real feature map gives
$88.7 \pm 3.6$ against a complex feature map's $75.0 \pm 3.1$. At thirty-two
parameters the same substitution in the feature map costs
$99.0 \pm 2.0 \to 91.3 \pm 4.5$. Making both the feature map and the ansatz
complex is worse again, at $86.5 \pm 5.3$. The training curves of
Figure~\ref{fig:qnntraining} locate the deficit in fitting rather than in
generalisation.

One cell dissents. With the weak rotation ansatz at ten parameters, a complex
feature map reaches $92.5 \pm 3.0$ against the real feature map's
$85.0 \pm 3.7$. It is the only comparison in the table favouring a complex
coin, it occurs at the smallest parameter count and with the least expressive
ansatz, and it runs against the other five. The reading consistent with the
rest of the evidence is that a rotation ansatz on the walk register is too
weak to exploit the real-coin encoding, and the complex encoding's extra
structure partly compensates; it is not support for the axis. Stated
plainly: on this task, axis~2 does not survive its controls, and the
contribution of this thesis rests on coin design and on the structural prior
rather than on complex amplitudes.

\paragraph{The task does not separate quantum from classical.}
A support vector machine with a radial-basis-function kernel, fitted to the
eight raw pixel values with default hyperparameters, classifies the test set
with $100$ per cent accuracy. Logistic regression reaches $65.0$, a linear
support vector machine $71.7$, and a majority-class predictor $53.3$ per
cent. The task is thus not linearly separable, which is why it is a
reasonable teaching example, but it is solved outright by a standard
classical kernel machine. Neither the course's $99$--$100$ per cent nor the
walk model's $99$ per cent is evidence of any quantum advantage, and none is
claimed. What the experiment establishes is a comparison among quantum
architectures --- which encoding carries the structure, and at what cost in
qubits and depth --- on a problem where the classical ceiling is known and
has already been reached.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_qnn_accuracy.pdf}
  \caption[Line-detection accuracy across architectures]{Test accuracy on the
  line-detection task, mean and standard deviation over ten weight
  initialisations. Bars are annotated with qubit and parameter counts. The
  rule marks the accuracy of a radial-basis-function support vector machine
  fitted to the raw pixels.}
  \label{fig:qnnaccuracy}
\end{figure}

\input{tab_qnn_results.tex}""",
    "5.2 QNN results")


io.open(TEX, "w", encoding="utf-8").write(s)
print("tex:", ", ".join(done))
print(f"tex bytes {orig_len} -> {len(s)}")


# ---------------------------------------------------------------- bib -----

b = io.open(BIB, encoding="utf-8").read()
add = []

if "IBMQML2026" not in b:
    add.append("""@misc{IBMQML2026,
  title       = {{Quantum Machine Learning}},
  author      = {{IBM Quantum}},
  year        = {2026},
  howpublished = {IBM Quantum Learning course},
  note        = {\\url{https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning}},
  urldate     = {2026-09-19}
}""")

if "McClean_2018" not in b:
    add.append("""@article{McClean_2018,
  title       = {{Barren plateaus in quantum neural network training landscapes}},
  author      = {Jarrod R. McClean and Sergio Boixo and Vadim N. Smelyanskiy and Ryan Babbush and Hartmut Neven},
  year        = {2018},
  journal     = {Nature Communications},
  volume      = {9},
  number      = {1},
  pages       = {4812},
  doi         = {10.1038/s41467-018-07090-4}
}""")

if "Havlicek_2019" not in b:
    add.append("""@article{Havlicek_2019,
  title       = {{Supervised learning with quantum-enhanced feature spaces}},
  author      = {Vojt{\\v{e}}ch Havl{\\'i}{\\v{c}}ek and Antonio D. C{\\'o}rcoles and Kristan Temme and Aram W. Harrow and Abhinav Kandala and Jerry M. Chow and Jay M. Gambetta},
  year        = {2019},
  journal     = {Nature},
  volume      = {567},
  number      = {7747},
  pages       = {209--212},
  doi         = {10.1038/s41586-019-0980-2}
}""")

if add:
    if not b.endswith("\n"):
        b += "\n"
    b += "\n" + "\n\n".join(add) + "\n"
    io.open(BIB, "w", encoding="utf-8").write(b)
    print(f"bib: added {len(add)} entries")
else:
    print("bib: nothing to add")
