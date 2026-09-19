# -*- coding: utf-8 -*-
"""Splice the kernel methodology (4.5) and the kernel results (5.3)."""

import io, os, sys

TEX = os.path.expanduser("~/mnt/thesis/draft.tex")
s = io.open(TEX, encoding="utf-8").read()
n0 = len(s)
done = []


def rep(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        print(f"FAIL [{tag}]: {n} occurrences"); sys.exit(1)
    s = s.replace(old, new); done.append(tag)


# ======================================================== 4.5 methodology
rep(
r"""\section{Application II---Quantum Kernel Methods}
\label{sec:kernelmethod}
\subsection{Analysis of the Existing Walk-Based Graph Kernel}
\subsection{Substitution of the Modified Walk}
\subsection{Positive Semi-Definiteness and Evaluation Cost}
\subsection{Datasets, Baselines and Evaluation Protocol}""",
r"""\section{Application II---Quantum Kernel Methods}
\label{sec:kernelmethod}

\subsection{The Quantum Kernel and the Role of the Feature Map}

A quantum kernel evaluates the inner product of two data points after they
have been encoded as quantum states~\cite{Schuld_2018,Havlicek_2019}. With a
feature map $U_\Phi$ and $\ket{\Phi(x)} = U_\Phi(x)\ket{0}^{\otimes n}$, the
fidelity kernel is
\begin{equation}
  K(x, y) \;=\; \bigl| \braket{\Phi(y) | \Phi(x)} \bigr|^2
  \;=\; \bigl| \bra{0}^{\otimes n} U_\Phi^\dagger(y)\, U_\Phi(x)
  \ket{0}^{\otimes n} \bigr|^2 ,
  \label{eq:fidkernel}
\end{equation}
which is the probability of observing the all-zero outcome after running
$U_\Phi(x)$ followed by $U_\Phi^\dagger(y)$. That circuit is drawn in
Figure~\ref{fig:circkernel}. The kernel matrix is then handed to a classical
support vector machine as a precomputed Gram matrix, so the only quantum
object in the pipeline is the feature map, and a comparison between kernels
is a comparison between encodings.

Equation~\eqref{eq:fidkernel} is positive semi-definite by construction: it is
a Gram matrix of unit vectors under the modulus-squared inner product.
Section~\ref{sec:kernelresults} verifies this numerically rather than
assuming it, since a violation beyond numerical tolerance would indicate an
implementation error rather than a mathematical surprise.

\subsection{The Walk as a Feature Map}
\label{sec:walkfmap}

The feature map used here is the plain coined walk on a cycle developed in
Chapter~\ref{ch:prelim}, with no modification to the walk itself. The
position register holds $d$ qubits and is read as a site in
$\mathbb{Z}_{2^d}$, where $d$ is the number of features; one further qubit
carries the coin. After preparing the uniform superposition with a layer of
Hadamards, each layer applies
\begin{equation}
  U_\Phi(x) \;=\; \Bigl[\, S \,\bigl(C \otimes I\bigr)\, D(x) \,\Bigr]^{L},
  \qquad
  D(x) \;=\; \bigotimes_{j=1}^{d} P\bigl(2 \lambda x_j\bigr),
  \label{eq:walkkernelmap}
\end{equation}
with $C$ the Hadamard coin, $S$ the shift
$\ket{0}\ket{p} \mapsto \ket{0}\ket{p-1}$,
$\ket{1}\ket{p} \mapsto \ket{1}\ket{p+1}$ modulo $2^d$, and $\lambda$ a
bandwidth. The circuit is Figure~\ref{fig:circwalkfm}, and the shift's
internal structure Figure~\ref{fig:circwalkshift}.

The design rests on one observation. The encoding layer $D(x)$ is, on its own,
exactly the $Z$ feature map: a product of single-qubit phases, separable,
incapable of representing any interaction between features. What couples the
features is the shift. Because $S$ increments the position register read as
an integer, it propagates through the carry chain and mixes every qubit with
every other; after one layer the amplitude at a site carries phase
contributions from its neighbours, and after $L$ layers from a widening
neighbourhood. The walk therefore supplies feature interaction through an
arithmetic operation, where the $ZZ$ feature map installs it as an explicit
layer of $O(d^2)$ two-qubit rotations. The two costs are of the same order;
the structures are not the same.

One design choice was made the wrong way first and is recorded because the
failure is instructive. An earlier version placed the walk on a cycle of
length $d$ --- one site per feature --- needing only
$\lceil \log_2 d \rceil + 1$ qubits. It is far cheaper and it does not work:
the feature space collapses to $2d$ dimensions, the fidelity kernel goes
flat, and classification falls to chance on every dataset tested. A feature
map must map into a space large enough to separate the data, and a
compression that looks like an efficiency is a loss of expressivity. The
position register is therefore one qubit per feature, and the walk map costs
one qubit \emph{more} than the $ZZ$ map rather than exponentially fewer.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_circ_walk_fm.pdf}
  \caption[The walk feature map as a circuit]{One layer of the walk feature
  map at four features, on five qubits. The phase gates $P(2\lambda x_j)$ are
  the encoding $D(x)$ and are separable; the Hadamard on the coin is $C$; the
  remaining gates are the shift, whose open and filled controls select the
  decrement and increment branches.}
  \label{fig:circwalkfm}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_circ_walk_shift.pdf}
  \caption[The shift operator as two ripple increments]{The shift alone. Each
  branch is a ripple of multiply-controlled $X$ gates implementing
  $p \mapsto p \pm 1$ modulo $2^d$; the carry out of the top qubit is
  discarded, which is precisely the cycle's wraparound. The carry chain is
  what couples features that the encoding layer leaves separable.}
  \label{fig:circwalkshift}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_circ_kernel.pdf}
  \caption[The fidelity kernel overlap circuit]{The overlap circuit for one
  kernel entry: $U_\Phi(x)$ followed by $U_\Phi^\dagger(y)$, with
  $K(x,y)$ read off as the probability of the all-zero measurement outcome.}
  \label{fig:circkernel}
\end{figure}

\subsection{Positive Semi-Definiteness and Evaluation Cost}

Building the Gram matrix requires $O(M^2)$ circuit evaluations for $M$
training points, which is the standard and unavoidable cost of a fidelity
kernel and is independent of the choice of map. Per evaluation, the walk map
costs $d$ single-qubit phase gates, $L$ coin gates and $2L$ controlled ripple
increments, the last at $O(d^2)$ two-qubit gates --- the same order as the
$ZZ$ map's entangling layer, on one more qubit.

The property that distinguishes the maps in practice is not gate count but
\emph{concentration}. Thanasilp et al.~\cite{Thanasilp_2024} show that
expressive quantum kernels tend to concentrate exponentially: as the number
of qubits grows, off-diagonal entries collapse towards zero, the Gram matrix
approaches the identity, and the model loses its ability to generalise
however many shots are spent estimating it. The mean off-diagonal entry of
the Gram matrix is a direct diagnostic for this, and is reported alongside
accuracy in Section~\ref{sec:kernelresults}.

\subsection{Datasets, Baselines and Evaluation Protocol}
\label{sec:kernelprotocol}

Five binary classification problems are used. The \emph{ad hoc} dataset of
Havlí\v{c}ek et al.~\cite{Havlicek_2019} is included because it is the
standard benchmark on which quantum kernels are claimed to separate data that
classical kernels do not, and because it was constructed around a
$ZZ$-type feature map rather than around a walk. The remaining four are
standard classification problems: iris restricted to versicolor against
virginica, which is the non-trivial pair; wine classes zero and one; the
Wisconsin breast cancer data; and handwritten digits three against eight.
Features are standardised and reduced by principal component analysis to
four or eight components as required, then standardised again.

The quantum baselines are the $Z$ and $ZZ$ feature maps as implemented in
Qiskit's circuit library, verified against it to $10^{-16}$. The classical
baselines are support vector machines with radial-basis-function and linear
kernels. Every model, quantum and classical, has its hyperparameters selected
by three-fold cross-validation on the training fold alone: the bandwidth
$\lambda$ and the encoding depth for the quantum maps, $\gamma$ for the
radial basis function, and the penalty $C$ for all of them, over grids of the
same size. Tuning one map and not another would measure the tuning rather
than the map. Each configuration is evaluated on five stratified
seventy--thirty splits, and results are reported as a mean and standard
deviation over those splits.""",
    "4.5 kernel methodology")


# ============================================================ 5.3 results
rep(
r"""\section{Quantum Kernel Methods}
\subsection{Kernel Matrices and Spectra}
\subsection{Classification Results}""",
r"""\section{Quantum Kernel Methods}
\label{sec:kernelresults}

\subsection{Kernel Matrices and Spectra}

Every Gram matrix produced in this study is positive semi-definite to
numerical precision. Across all maps, datasets and splits the most negative
eigenvalue observed is $-3.1 \times 10^{-14}$, which is rounding error on a
matrix of unit-trace entries. Equation~\eqref{eq:fidkernel} guarantees this,
and the check confirms the implementation rather than the mathematics.

The spectra are more informative about what separates the maps. The mean
off-diagonal entry of the training Gram matrix, reported in
Table~\ref{tab:kernel}, measures how far the kernel is from the identity, and
it is where the $ZZ$ map fails. On the two four-feature problems its mean
off-diagonal entry is $0.35$ and $0.28$; on the two eight-feature problems it
falls to $0.10$ and $0.097$, and the smallest eigenvalue rises to $0.12$ and
$0.26$ --- a Gram matrix that has become nearly diagonal. This is exactly the
exponential concentration Thanasilp et al.~\cite{Thanasilp_2024} describe:
the map is expressive enough that distinct inputs become nearly orthogonal,
and the kernel stops carrying information about similarity.

The walk map does not concentrate in the same way. At eight features its mean
off-diagonal entries are $0.69$ and $0.60$, close to the $Z$ map's $0.68$ and
$0.60$ and far from the $ZZ$ map's. The walk introduces feature interaction
through the carry chain of an increment rather than through a layer of
pairwise $ZZ$ rotations, and on this evidence that mechanism buys the
interaction without driving the kernel to the identity. This is the clearest
mechanistic result in the chapter, and it is what the accuracies below
reflect.

\subsection{Classification Results}

Table~\ref{tab:kernel} and Figure~\ref{fig:kernelacc} give test accuracy for
every kernel on every dataset.

\paragraph*{The walk kernel is the strongest of the three quantum kernels.}
It is the best quantum kernel on three of the five datasets --- ad hoc
($82.6 \pm 4.3$ against the $Z$ map's $80.7 \pm 3.8$), wine
($95.9 \pm 4.2$ against $94.9 \pm 3.6$) and digits ($97.7 \pm 1.3$ against
$97.0 \pm 1.6$) --- and ties the $Z$ map on the other two. It beats the $ZZ$
map on all five, by margins from $2.2$ to $5.6$ points. The margin over the
$Z$ map is small and within one standard deviation on each individual
dataset; the consistency of its direction across five datasets is a better
guide than any single comparison, and the honest statement is that the walk
map is at least as good as the best conventional map everywhere and clearly
better than the entangling one.

\paragraph*{On the benchmark built for quantum kernels, it is the best
method of any kind.} On the ad hoc dataset the walk kernel reaches
$82.6 \pm 4.3$ per cent against $80.7$ for the $Z$ map, $77.0$ for the
$ZZ$ map, $73.3$ for a tuned radial-basis-function machine and $54.1$ for a
linear one. This is the one dataset in the study where every quantum kernel
beats every classical kernel, which is what it was constructed to
exhibit~\cite{Havlicek_2019}, and the walk map exhibits it most strongly ---
on a dataset designed around a $ZZ$-type encoding rather than around a walk.

\paragraph*{On ordinary data the classical kernels remain competitive.}
On the four standard problems the best classical baseline is within a point
of the best quantum kernel every time, and on breast cancer a linear support
vector machine is the outright best method at $96.0 \pm 2.9$. No claim of
quantum advantage is made or supported here. What the results establish is
the narrower and better-founded claim: as an encoding, a plain coined
quantum walk is a better fidelity kernel than the standard entangling
feature map, it resists the concentration that degrades that map at eight
features, and it costs gates of the same order on one additional qubit.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_kernel_acc.pdf}
  \caption[Quantum kernel accuracy across datasets]{Test accuracy of the
  fidelity kernels and classical baselines, mean and standard deviation over
  five stratified splits. All hyperparameters are tuned by cross-validation
  on the training fold only.}
  \label{fig:kernelacc}
\end{figure}

\input{tab_kernel.tex}""",
    "5.3 kernel results")


io.open(TEX, "w", encoding="utf-8").write(s)
print("applied:", ", ".join(done))
print(f"bytes {n0} -> {len(s)}")
