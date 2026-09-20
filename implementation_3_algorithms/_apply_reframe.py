import sys
p = 'draft.tex'
s = open(p, encoding='utf-8').read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)


# ---------- 1. cut the classical ML overview (Sec. 2.2) ----------
i = s.index('\\section{An Overview of Machine Learning}')
j = s.index('\\section{Quantum Walks}\n\\label{sec:prelim-qw}')
removed = s[i:j]
s = s[:i] + s[j:]
open('../code/implementation_3_algorithms/_ml_overview_removed.tex', 'w', encoding='utf-8').write(removed)

rep("""quantum walk is measured against; Section~\\ref{sec:prelim-ml} reviews
classical machine learning at the level needed to read the learning models
of Chapter~\\ref{ch:methodology}, and ends with two worked classical
implementations; Section~\\ref{sec:prelim-qw} develops""",
    """quantum walk is measured against; Section~\\ref{sec:prelim-qw} develops""")
rep("""% skeletons (headings + scope notes); 2.2 is the former 3.3 moved. The""",
    """% skeletons (headings + scope notes). The classical machine-learning
% overview that stood at 2.2 was cut entirely on the author's instruction
% (20 Sep 2026): classical ML is reviewed in Sec. 1.3 and nowhere else. Its
% text is in draft.tex.bak-pre-mlcut and in
% code/implementation_3_algorithms/_ml_overview_removed.tex. The""")

# ---------- 2. appendix: remove references into the cut section ----------
rep("""This appendix collects two complete classical implementations,
referred to from Section~
ef{sec:ml-graphs}, for two reasons.""",
    """This appendix collects two complete classical implementations, for two
reasons.""")
rep("""It is the simplest instance of~\\eqref{eq:mlp}, with one learned
layer on either side of the pooling.""",
    """It is the simplest instance of a feed-forward network, with one
learned layer on either side of the pooling.""")
rep("""The second listing trains a fully connected network of the
form~\\eqref{eq:mlp}, with two hidden layers""",
    """The second listing trains a fully connected feed-forward network,
with two hidden layers""")

# ---------- 3. delete Sec. 4.1 (empty; duplicates 4.3) ----------
rep("""\\section{The Quantum Walk as a Learning Primitive}
\\subsection{Where a Walk Enters a Learning Algorithm}
\\subsection{Sampling, Similarity and Feature Construction}

""", "")

# ---------- 4. reframe the gap ----------
k = s.index('\\subsection{The Gap Addressed by This Thesis}')
e = s.index('\\section{Problem Statement}')
gap = open('/home/claude/impl3/tex/gap_reframed.tex', encoding='utf-8').read() if False else None
gap = open('_gap_reframed.tex', encoding='utf-8').read()
s = s[:k] + gap + s[e:]

# ---------- 5. problem-statement skeleton ----------
old_ps = s[s.index('\\section{Problem Statement}'):s.index('\\section{The Structure of This Thesis}')]
new_ps = open('_problem_statement_skeleton.tex', encoding='utf-8').read()
s = s.replace(old_ps, new_ps)

# ---------- 6. preamble scope block ----------
rep("""%  Scope of the contribution (spec rewritten 20 Sep 2026):
%    * The walk is modified along TWO axes -- the coin read as a design space
%      (weights that may depend on local graph structure and on the step
%      index), and complex rather than real amplitudes. The memory register
%      is NOT a contribution axis of this thesis; do not reintroduce it.""",
    """%  Scope of the contribution (author, 20 Sep 2026 -- SUPERSEDES the earlier
%  coin-centred framing):
%    * The subject is the QUANTUM WALK AS A GENERAL ALGORITHMIC TOOL. The
%      same primitive is exercised across algorithm families -- order
%      finding, QAOA and search (Ch. 3), quantum neural networks,
%      variational classifiers and quantum kernels (Chs. 4-5) -- and for
%      each the thesis asks what the walk formulation is, what it costs as a
%      circuit, and what it achieves against a tuned baseline.
%    * Design choices inside the walk (coin weights depending on local graph
%      structure or on the step index; complex rather than real amplitudes;
%      flip-flop vs moving shift) are VARIABLES examined within that
%      programme, not the contribution. Do not reframe the thesis around the
%      coin. The memory register is not an axis at all; do not reintroduce
%      it.""")
rep("""%    * RESEARCH QUESTION (author, 20 Sep 2026), which is the problem framing
%      and NOT a change of title -- the title stays "Quantum Walks as a Tool
%      for Quantum Machine Learning Algorithms":
%        Quantum Walk-Based Quantum Neural Networks and Quantum Kernels for
%        Graph-Structured Data: Design, Trainability, and Empirical Evaluation
%      Two families (QNN/VQC and kernels), graph-structured data as the
%      domain, and three axes: design, trainability, empirical evaluation.
%      See the note at Sec. 1.3.""",
    """%    * The "Design, Trainability, and Empirical Evaluation" research
%      question drafted earlier on 20 Sep 2026 was NOT adopted; the title
%      stays "Quantum Walks as a Tool for Quantum Machine Learning
%      Algorithms" and the framing is the one above.""")

open(p, 'w', encoding='utf-8').write(s)
print('ok; ML words removed:', len(removed.split()))
