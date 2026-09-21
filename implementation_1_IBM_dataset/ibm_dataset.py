"""
The line-detection dataset from IBM's Quantum Machine Learning course, lesson
"QVCs and QNNs".
"""

from __future__ import annotations

import numpy as np

V_DIM, H_DIM = 2, 4
N_PIXELS = V_DIM * H_DIM
SIGNAL = np.pi / 2
NOISE_HI = np.pi / 4

LABEL_HORIZONTAL = -1
LABEL_VERTICAL = +1


def horizontal_pairs() -> list[tuple[int, int]]:
    """Within-row adjacent pixel pairs -- also the course's CNOT list."""
    return [(r * H_DIM + c, r * H_DIM + c + 1)
            for r in range(V_DIM) for c in range(H_DIM - 1)]


def vertical_pairs() -> list[tuple[int, int]]:
    """Within-column adjacent pixel pairs."""
    return [(r * H_DIM + c, (r + 1) * H_DIM + c)
            for r in range(V_DIM - 1) for c in range(H_DIM)]


def grid_edges() -> list[tuple[int, int]]:
    """All 4-neighbour adjacencies of the pixel grid."""
    return horizontal_pairs() + vertical_pairs()


def make_dataset(n_images: int = 200, seed: int = 42):
    """Returns X of shape (n_images, 8) and y in {-1, +1}."""
    rng = np.random.default_rng(seed)
    hp, vp = horizontal_pairs(), vertical_pairs()

    X = rng.uniform(0.0, NOISE_HI, size=(n_images, N_PIXELS))
    y = np.empty(n_images, dtype=int)

    for i in range(n_images):
        if rng.integers(2) == 0:
            a, b = hp[rng.integers(len(hp))]
            y[i] = LABEL_HORIZONTAL
        else:
            a, b = vp[rng.integers(len(vp))]
            y[i] = LABEL_VERTICAL
        X[i, a] = SIGNAL
        X[i, b] = SIGNAL

    return X, y


def train_test_split(X, y, test_fraction: float = 0.3, seed: int = 246):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(round(test_fraction * len(y)))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def load(n_images: int = 200, seed: int = 42, split_seed: int = 246):
    X, y = make_dataset(n_images, seed)
    return train_test_split(X, y, seed=split_seed)


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    print(f"train {Xtr.shape}  test {Xte.shape}")
    print(f"train class balance: {np.bincount((ytr + 1) // 2)}  (horizontal, vertical)")
    print(f"test  class balance: {np.bincount((yte + 1) // 2)}")
    print(f"horizontal pairs: {horizontal_pairs()}")
    print(f"vertical pairs  : {vertical_pairs()}")
    print(f"pixel range: [{Xtr.min():.4f}, {Xtr.max():.4f}]  signal = {SIGNAL:.4f}")
    print("\nfirst horizontal image (2x4):")
    i = int(np.where(ytr == LABEL_HORIZONTAL)[0][0])
    print(np.round(Xtr[i].reshape(V_DIM, H_DIM), 3))
    print("first vertical image (2x4):")
    i = int(np.where(ytr == LABEL_VERTICAL)[0][0])
    print(np.round(Xtr[i].reshape(V_DIM, H_DIM), 3))
