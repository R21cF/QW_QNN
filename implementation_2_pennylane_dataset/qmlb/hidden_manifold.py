# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0


import numpy as np


def neural_net(x, W, v):
    """
    Transforms inputs via a single-layer neural network. Args: x (ndarray):
    input of shape (manifold_dimension,) W (ndarray): input-to-hidden weight
    matrix of shape (manifold_dimension, manifold_dimension) v (ndarray):
    hidden-to-output weight matrix of shape (manifold_dimension,)
    """
    return np.dot(v, np.tanh(W @ x) / np.sqrt(W.shape[0]))


def nonlinearity(X, biases):
    """Element-wise nonlinearity."""
    return np.tanh(X - biases)


def generate_hidden_manifold_model(n_samples, n_features, manifold_dimension):
    """Data generation procedure for the 'hidden manifold model'."""

    # feature matrix F controls the embedding of the manifold
    F = np.random.normal(size=(manifold_dimension, n_features))

    # Gaussian matrix samples original inputs from the lower-dimensional manifold
    C = np.random.normal(size=(n_samples, manifold_dimension), loc=0, scale=1)

    # embed data, adding an element-wise nonlinearity
    biases = 2 * np.random.uniform(size=(n_features,)) - 1
    X = nonlinearity(C @ F / np.sqrt(manifold_dimension), biases)

    # define labels via a neural network
    W = np.random.normal(size=(manifold_dimension, manifold_dimension))
    v = np.random.normal(size=(manifold_dimension,))
    y = np.array([neural_net(c, W, v) for c in C])

    # post-process the labels to get balanced classes
    y = y - np.median(y)
    y = np.array([-1 if y_ < 0 else 1 for y_ in y])
    assert len(X[y == 1]) == n_samples // 2
    assert len(X[y == -1]) == n_samples // 2

    return X, y
