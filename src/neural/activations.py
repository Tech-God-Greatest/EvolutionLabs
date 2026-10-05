"""Activation functions (NumPy)."""
from __future__ import annotations

import numpy as np


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def relu(x):
    return np.maximum(x, 0.0)


def identity(x):
    return x


ACTIVATIONS = {"tanh": tanh, "sigmoid": sigmoid, "relu": relu, "identity": identity}
