# -*- coding: utf-8 -*-
"""Splice the efficient-construction methodology and its measured scaling."""

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


# ==================================================================== 4.3.4.3
rep(
"""\\subsubsection{Circuit Construction and Resource Cost}
\\label{sec:efficientcircuit}
% Focus on effects of using quantum walks instead of other frameworks
% TODO: the efficient (non-dense-synthesis) walk-step construction goes
% here.  Sec.~5.1.3 forward-references this label.""",
r"""\subsubsection{Circuit Construction and Resource Cost}
\label{sec:efficientcircuit}
% Focus on effects of using quantum walks instead of other frameworks

The general construction of Section~\ref{sec:qiskittooling} appends each walk
step as one dense unitary on all $n$ qubits. Generic synthesis of an
$n$-qubit unitary requires $\Theta(4^n)$ two-qubit gates, and
$n = \lceil \log_2 N \rceil + \lceil \log_2 d \rceil$, so the cost of a step
grows as $O(N^2 d^2)$ in the size of the graph. That is the origin of the
depths reported in Section~\ref{sec:depth}, and it is a property of the
synthesis rather than of the walk. This section asks what the walk step
actually costs when it is built rather than synthesised.

\paragraph*{Decomposition by edge colouring.}
Take a proper edge colouring of $G$, assigning colour $c$ to each edge so
that no vertex meets two edges of the same colour, and index the coin by
colour rather than by a per-vertex ordering of neighbours. Each colour class
is then a matching, and the shift acts within a colour:
\begin{equation}
  S \;=\; \sum_{c} \ket{c}\bra{c} \otimes M_c,
  \qquad
  M_c \ket{v} \;=\; \ket{m_c(v)},
  \label{eq:colourshift}
\end{equation}
where $m_c$ swaps the endpoints of every edge of colour $c$ and fixes any
vertex that meets no such edge. Because a matching pairs each vertex at most
once, $M_c$ is an involution and a permutation of the position register
alone; the coin index is untouched. By Vizing's theorem the number of terms
is at most $\Delta + 1$, and exactly $\Delta$ for a bipartite graph, so the
sum in~\eqref{eq:colourshift} grows with the \emph{degree} and not with the
number of vertices. This is the decomposition of Douglas and
Wang~\cite{Douglas_2007}, developed at length in Loke's
thesis~\cite{Han_2017}.

\paragraph*{Why the colouring alone is not enough.}
Equation~\eqref{eq:colourshift} replaces one $n$-qubit operator by $\Delta$
controlled permutations of the $\lceil \log_2 N \rceil$-qubit position
register. If each $M_c$ is handed to the compiler as a dense matrix, however,
it is still synthesised generically --- at $O(N^2)$ in the position register
--- and the coin controls make it worse rather than better. Measured, that
construction is \emph{slower} than the dense step beyond five qubits, by
roughly a factor of two on a twelve-vertex grid. The colouring is necessary
and not sufficient: what matters is whether $M_c$ has a description the
compiler can exploit.

\paragraph*{Colour classes with an arithmetic description.}
For structured graphs each $M_c$ is not an arbitrary matching but one that
can be written in arithmetic on the position register, and the cost collapses.

On the hypercube $Q_m$ the natural colouring assigns colour $i$ to every edge
that flips bit $i$, which is a perfect matching, so $M_i$ is a single $X$ gate
on position qubit $i$ and the entire shift is $m$ controlled-$X$ gates. This
is the construction underlying the walk-based search algorithm of Shenvi,
Kempe and Whaley.

On a cycle of length $2^m$, and on any torus formed as a product of such
cycles, the neighbours of a site are its coordinates $\pm 1$, so the shift is
a controlled increment or decrement of one coordinate register. An increment
modulo $2^m$ is the ripple
$\mathrm{MCX}(q_0 \ldots q_{k-1} \to q_k)$ for $k = m-1$ down to $1$,
followed by $X$ on $q_0$; the carry out of the top qubit is simply discarded,
which is exactly the wraparound the cycle requires. Decrement is the inverse
sequence. The whole shift is therefore $2r$ controlled ripples on a
$r$-dimensional torus, each $O(m^2)$ in two-qubit gates with $m = \log_2 L$,
giving a cost polylogarithmic in $N$.

\paragraph*{The coin is what remains, and it is the binding constraint.}
None of the above touches the coin, and the coin is where this thesis's first
axis lives. A coin that is the same at every vertex acts on the coin register
only: for $d \le 4$ it is a single two-qubit gate, at most three CX, and
contributes nothing to the scaling. A coin that varies with the vertex ---
whether by degree, by local structure or by data --- must be applied
conditioned on the position register holding that vertex, which is one
multi-controlled operation per vertex, with $\lceil \log_2 N \rceil$ controls
each. No colouring and no arithmetic description reduces that, because the
cost is not in the shift.

The measurements in Section~\ref{sec:depth} put the penalty at three orders of
magnitude on a thirty-two-vertex torus and growing as roughly $N^{3.5}$, which
is worse than the dense construction the whole exercise set out to avoid.
The conclusion is unwelcome but clear: a structure-dependent coin is not
implementable at any interesting scale, and a walk model intended for hardware
must either share one coin across all vertices or introduce its per-vertex
dependence somewhere cheaper. The natural cheaper place is a diagonal phase
on the position register, $D(x) = \operatorname{diag}(e^{i \alpha x_v})$,
which under the standard Walsh--Hadamard decomposition costs $N-1$ CX once
per encoding layer rather than $N$ multi-controlled coins per step. That
variant is not evaluated here and is left to future work.""",
    "4.3.4.3 efficient construction")


# ====================================================================== 5.1.3
rep(
r"""Section~\ref{sec:efficientcircuit}; the figures in this section stand as the
naive baseline that work has to beat.""",
r"""Section~\ref{sec:efficientcircuit}; the figures in this section stand as the
naive baseline that work has to beat.

\paragraph*{The structured construction beats it, and by a widening margin.}
Table~\ref{tab:scaling} and Figure~\ref{fig:walkscaling} compare the two
constructions on one walk step, over graph families for which the colour
classes admit the arithmetic description of
Section~\ref{sec:efficientcircuit}. Both constructions were verified against
the same operator before being costed: the structured circuit reproduces the
dense step matrix to better than $10^{-12}$ on every instance tested.

The dense step behaves as predicted. Fitting $\mathrm{CX} \sim N^{\alpha}$
over the instances that could be built gives $\alpha = 2.06$ for the cycle
and $\alpha = 2.02$ for the torus, against the $O(N^2 d^2)$ expected from
$\Theta(4^n)$ synthesis. Beyond about nine qubits the dense operator is not
worth constructing at all.

The structured step does not. Fitting over the larger half of each family,
where constant overheads no longer dominate, gives $\alpha = 0.44$ for the
cycle, $0.29$ for the torus and $0.19$ for the hypercube; a fit of the form
$\mathrm{CX} \sim (\log_2 N)^{\beta}$ describes the same data with
$\beta$ between $1.7$ and $3.1$. The step is polylogarithmic in the number of
vertices. In absolute terms the saving reaches a factor of $132$ on a
256-vertex cycle, $632$ on a 128-vertex torus and $1283$ on the 64-vertex
hypercube $Q_6$, and it grows with every doubling. A walk on a 1024-vertex
cycle costs $1614$ two-qubit gates per step under the structured
construction; the dense operator for it has $2^{22}$ entries and was not
built.

\paragraph*{The coin, not the shift, is now the bottleneck.}
That result removes the shift from the cost budget and leaves the coin
exposed. With a uniform coin the full step tracks the shift almost exactly:
$101$ CX against the shift's $98$ on a thirty-two-vertex torus, the coin
contributing three. With a coin that differs at every vertex the same step
costs $132{,}515$ CX --- a factor of $1312$ --- and the growth fits
$N^{3.54}$, steeper than the dense construction that motivated this work. The
$64$-vertex case could not be transpiled in reasonable time at all, which is
itself the clearest available statement of the problem.

Two conclusions follow for this thesis. The walk step is not intrinsically
expensive: built rather than synthesised, on a graph with exploitable
structure, it is cheap and stays cheap. But the structure-dependent coin of
axis~1, as implemented in Chapter~\ref{ch:methodology}, is not implementable
at any scale worth reaching, and no improvement to the shift changes that.
The results of Section~\ref{sec:qnnresults} are therefore simulation results
about an architecture, not a proposal that could be run on a device in its
present form.

A third conclusion is a limit on the first. Every family in
Table~\ref{tab:scaling} is a lattice, and the construction depends on the
colour classes having an arithmetic description. A molecular graph from a
classification benchmark has no such description, and for the graph kernels
of Section~\ref{sec:kernelmethod} the dense cost stands unimproved. The
saving reported here is real and is confined to structured graphs.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_walk_scaling.pdf}
  \caption[Circuit cost of a walk step against graph size]{Two-qubit gates
  per walk step after transpilation, against the number of vertices, on
  logarithmic axes. Dense synthesis grows as $N^2$; the structured shift with
  a uniform coin is polylogarithmic. The per-vertex coin of axis~1 grows as
  $N^{3.5}$ and dominates both.}
  \label{fig:walkscaling}
\end{figure}

\input{tab_scaling.tex}""",
    "5.1.3 scaling results")


# label for Chapter 4, referenced above
rep(r"""\chapter{QUANTUM-WALK-BASED ALGORITHMS FOR QUANTUM MACHINE LEARNING}""",
    "\\chapter{QUANTUM-WALK-BASED ALGORITHMS FOR QUANTUM MACHINE LEARNING}\n"
    "\\label{ch:methodology}", "label methodology")

io.open(TEX, "w", encoding="utf-8").write(s)
print("applied:", ", ".join(done))
print(f"bytes {n0} -> {len(s)}")
