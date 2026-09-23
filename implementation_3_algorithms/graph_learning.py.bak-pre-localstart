"""Walk-based kernels and a walk-based network for graph classification."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from common import walk_operator, uniform_arc_state, position_marginal

SEED = 3
N_PER_CLASS = 100
T = 8
REPEATS = 3


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_dataset(kind, rng):
    graphs, labels = [], []
    for _ in range(N_PER_CLASS):
        n = int(rng.choice([8, 10, 12]))
        if kind == "regular":
            G1 = nx.random_regular_graph(3, n, seed=int(rng.integers(1 << 30)))
            m = G1.number_of_edges()
            G0 = nx.gnm_random_graph(n, m, seed=int(rng.integers(1 << 30)))
        elif kind == "bipartite":
            m = int(1.5 * n)
            n1 = n // 2
            G1 = nx.bipartite.gnmk_random_graph(n1, n - n1, m, seed=int(rng.integers(1 << 30)))
            G0 = nx.gnm_random_graph(n, m, seed=int(rng.integers(1 << 30)))
        elif kind == "community":
            p_in, p_out = 0.7, 0.1
            sizes = [n // 2, n - n // 2]
            G1 = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]],
                                           seed=int(rng.integers(1 << 30)))
            p_avg = (p_in * (sizes[0] * (sizes[0] - 1) / 2 + sizes[1] * (sizes[1] - 1) / 2)
                     + p_out * sizes[0] * sizes[1]) / (n * (n - 1) / 2)
            G0 = nx.gnp_random_graph(n, p_avg, seed=int(rng.integers(1 << 30)))
        else:
            raise ValueError(kind)
        for G, y in ((G1, 1), (G0, 0)):
            G.remove_nodes_from(list(nx.isolates(G)))      # isolated vertices have no arcs
            G = nx.convert_node_labels_to_integers(G)
            if G.number_of_nodes() < 4:
                continue
            graphs.append(G)
            labels.append(y)
    return graphs, np.array(labels)


def load_tu(root, name):
    """
    TU-Dortmund format (Morris et al.): *_A.txt, *_graph_indicator.txt,
    *_graph_labels.txt, optional *_node_labels.txt. Node labels are stored as
    the 'label' attribute; the walk ignores them, the WL kernel uses them.
    """
    import os
    p = lambda suf: os.path.join(root, name, f"{name}_{suf}.txt")
    ind = np.loadtxt(p("graph_indicator"), dtype=int)
    gl = np.loadtxt(p("graph_labels"), dtype=int)
    A = np.loadtxt(p("A"), delimiter=",", dtype=int)
    nl = np.loadtxt(p("node_labels"), dtype=int) if os.path.exists(p("node_labels")) else None
    graphs = []
    for g in range(1, ind.max() + 1):
        nodes = np.where(ind == g)[0] + 1
        G = nx.Graph()
        for v in nodes:
            G.add_node(int(v), label=int(nl[v - 1]) if nl is not None else 1)
        graphs.append(G)
    gid = {}
    for k, G in enumerate(graphs):
        for v in G.nodes():
            gid[v] = k
    for u, v in A:
        graphs[gid[int(u)]].add_edge(int(u), int(v))
    out, ys = [], []
    for G, y in zip(graphs, gl):
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 4:
            continue
        out.append(nx.convert_node_labels_to_integers(G))
        ys.append(1 if y > 0 else 0)
    return out, np.array(ys)


def load_gin(root, name):
    """
    Format of Xu et al.'s GIN repository (weihua916/powerful-gnns/dataset): line
    1 = #graphs; per graph a header 'n y' then n lines 'label deg nb...'.
    """
    import os
    with open(os.path.join(root, name, f"{name}.txt")) as f:
        ng = int(f.readline())
        out, ys = [], []
        for _ in range(ng):
            n, y = map(int, f.readline().split())
            G = nx.Graph()
            rows = [list(map(int, f.readline().split())) for _ in range(n)]
            for v, r in enumerate(rows):
                G.add_node(v, label=r[0])
            for v, r in enumerate(rows):
                for u in r[2:2 + r[1]]:
                    if u != v:
                        G.add_edge(v, u)
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() < 4:
                continue
            out.append(nx.convert_node_labels_to_integers(G))
            ys.append(y)
    ys = np.array(ys)
    return out, (ys == ys.max()).astype(int) if len(set(ys)) == 2 else ys


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------
def _stats(p):
    n = len(p)
    p = np.clip(p, 1e-15, None)
    return [float(-(p * np.log(p)).sum()), float((p ** 2).sum() * n), float(p.max() * n)]


class WalkSignature:
    """Precomputes the walk as index maps so that trainable-coin sweeps are cheap."""

    def __init__(self, G):
        self.G = G
        self.n = G.number_of_nodes()
        self.nodes = list(G.nodes())
        self.deg = np.array([G.degree(v) for v in self.nodes])
        self.d = int(self.deg.max())
        idx = {v: i for i, v in enumerate(self.nodes)}
        nbrs = [list(G.neighbors(v)) for v in self.nodes]
        self.src = []; self.dst = []
        for i, v in enumerate(self.nodes):
            for a, u in enumerate(nbrs[i]):
                b = nbrs[idx[u]].index(v)
                self.src.append(i * self.d + a); self.dst.append(idx[u] * self.d + b)
        self.src = np.array(self.src); self.dst = np.array(self.dst)
        self.mask = np.zeros((self.n, self.d))
        for i in range(self.n):
            self.mask[i, :self.deg[i]] = 1.0
        B = self.n + 1
        psi0 = np.zeros((B, self.n, self.d), dtype=complex)
        psi0[0] = self.mask
        for v in range(self.n):
            psi0[v + 1, v] = self.mask[v]
        psi0 = psi0.reshape(B, -1)
        self.psi0 = psi0 / np.linalg.norm(psi0, axis=1, keepdims=True)

    def grover(self, psi):
        m = psi.reshape(-1, self.n, self.d)
        s = (m * self.mask).sum(axis=2, keepdims=True)
        g = (2.0 / np.maximum(self.deg, 1))[None, :, None] * s - m
        return (g * self.mask + m * (1 - self.mask)).reshape(psi.shape[0], -1)

    def step(self, psi, theta):
        c = np.cos(theta) * psi - 1j * np.sin(theta) * self.grover(psi)
        out = c.copy()
        out[:, self.dst] = c[:, self.src]
        return out

    def run(self, thetas):
        psi = self.psi0
        feats = []
        for th in thetas:
            psi = self.step(psi, th)
            P = (np.abs(psi.reshape(-1, self.n, self.d)) ** 2).sum(axis=2)   # (B, n)
            feats += _stats(P[0])
            feats.append(float(np.mean([P[v + 1, v] for v in range(self.n)])))
        return np.array(feats)


def classical_signature(G, T):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    P = 0.5 * np.eye(n) + 0.5 * A / A.sum(axis=1, keepdims=True)
    p = np.ones(n) / n
    Pt = np.eye(n)
    feats = []
    for _ in range(T):
        p = p @ P
        Pt = Pt @ P
        feats += _stats(p)
        feats.append(float(np.trace(Pt) / n))
    return np.array(feats)


def handmade(G):
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()])
    hist = np.bincount(deg, minlength=8)[:8] / n
    lam = np.linalg.eigvalsh(nx.to_numpy_array(G))
    return np.concatenate([hist, [nx.average_clustering(G), lam[-1], lam[0], nx.density(G)]])


def wl_kernel(graphs, h=3):
    """Weisfeiler-Lehman subtree kernel (Shervashidze et al. 2011), normalised."""
    labels = [{v: G.nodes[v].get("label", 1) for v in G.nodes()} for G in graphs]
    feats = [Counter() for _ in graphs]
    lookup = {}
    for it in range(h + 1):
        for gi, G in enumerate(graphs):
            for v, l in labels[gi].items():
                feats[gi][(it, l)] += 1
        if it == h:
            break
        new = []
        for gi, G in enumerate(graphs):
            nl = {}
            for v in G.nodes():
                key = (labels[gi][v], tuple(sorted(labels[gi][u] for u in G.neighbors(v))))
                if key not in lookup:
                    lookup[key] = len(lookup) + 2
                nl[v] = lookup[key]
            new.append(nl)
        labels = new
    keys = sorted({k for f in feats for k in f})
    X = np.array([[f[k] for k in keys] for f in feats], float)
    K = X @ X.T
    dg = np.sqrt(np.diag(K))
    return K / np.outer(dg, dg)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def cv_svm_features(X, y, rng):
    grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    accs = []
    for rep in range(REPEATS):
        skf = StratifiedKFold(10, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = GridSearchCV(SVC(kernel="rbf"), grid, cv=3).fit(sc.transform(X[tr]), y[tr])
            accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def cv_svm_precomputed(K, y):
    accs = []
    for rep in range(REPEATS):
        skf = StratifiedKFold(10, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(K, y):
            clf = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]}, cv=3)
            clf.fit(K[np.ix_(tr, tr)], y[tr])
            accs.append(clf.score(K[np.ix_(te, tr)], y[te]))
    return float(np.mean(accs)), float(np.std(accs))


def cv_walk_network(sigs, y, rng, trainable=True, maxiter=60):
    """Trainable coins + logistic read-out on the final-step statistics."""
    accs = []
    n_feat = 4 * T

    def features(thetas):
        return np.array([s.run(thetas) for s in sigs])

    for rep in range(REPEATS):
        skf = StratifiedKFold(10, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            if trainable:
                def loss(params):
                    th = params[:T]
                    X = features(th)[tr]
                    sc = StandardScaler().fit(X)
                    lr = LogisticRegression(C=1.0, max_iter=200).fit(sc.transform(X), y[tr])
                    p = lr.predict_proba(sc.transform(X))[:, 1]
                    return float(-np.mean(y[tr] * np.log(p + 1e-12) + (1 - y[tr]) * np.log(1 - p + 1e-12)))
                th0 = np.full(T, np.pi / 2) + rng.normal(0, 0.2, T)
                res = minimize(loss, th0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
                th = res.x[:T]
            else:
                th = np.full(T, np.pi / 2)
            X = features(th)
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=1.0, max_iter=200).fit(sc.transform(X[tr]), y[tr])
            accs.append(lr.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    out = []
    BENCH = {"MUTAG": ("tu", "MUTAG"), "PTC_MR": ("gin", "PTC"),
             "PROTEINS": ("gin", "PROTEINS"), "IMDB-BINARY": ("gin", "IMDBBINARY")}
    NETWORK_ON = {"regular", "bipartite", "community", "MUTAG", "PTC_MR"}   # walk network is COBYLA-trained; skipped on the two large sets
    for kind in ["regular", "bipartite", "community", "MUTAG", "PTC_MR", "PROTEINS", "IMDB-BINARY"]:
        if kind in BENCH:
            fmt, nm = BENCH[kind]
            graphs, y = load_tu("data", nm) if fmt == "tu" else load_gin("data/gin/dataset", nm)
        else:
            graphs, y = make_dataset(kind, rng)
        sigs = [WalkSignature(G) for G in graphs]
        Xq = np.array([s.run([np.pi / 2] * T) for s in sigs])
        # cross-check the fast walk against the dense operator on one graph
        G0 = graphs[0]
        W, nodes, d = walk_operator(G0)
        psi = uniform_arc_state(G0, nodes, d)
        ref = []
        for _ in range(T):
            psi = W @ psi
            ref += _stats(position_marginal(psi, len(nodes), d))
        # Grover coin = -i * exp(-i pi/2 G) up to a global phase; stats agree
        assert np.allclose(ref, Xq[0].reshape(T, 4)[:, :3].reshape(-1), atol=1e-8), \
            "fast walk disagrees with dense operator"
        Xc = np.array([classical_signature(G, T) for G in graphs])
        Xh = np.array([handmade(G) for G in graphs])
        K_wl = wl_kernel(graphs)
        row = {"task": kind, "n_graphs": len(graphs), "sizes": sorted(set(G.number_of_nodes() for G in graphs)),
               "class_balance": float(y.mean())}
        row["DTQW signature kernel"] = cv_svm_features(Xq, y, rng)
        row["classical RW signature kernel"] = cv_svm_features(Xc, y, rng)
        row["WL subtree kernel"] = cv_svm_precomputed(K_wl, y)
        row["hand-made statistics"] = cv_svm_features(Xh, y, rng)
        if kind in NETWORK_ON:
            row["walk network (fixed Grover coin)"] = cv_walk_network(sigs, y, rng, trainable=False)
            row["walk network (trained coins)"] = cv_walk_network(sigs, y, rng, trainable=True)
        out.append(row)
        for k_, v in row.items():
            if isinstance(v, tuple):
                print(f"{kind:10s} {k_:36s} {100*v[0]:5.1f} +- {100*v[1]:4.1f}")
    json.dump(out, open("graph_results.json", "w"), indent=1)
