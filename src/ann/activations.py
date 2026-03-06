"""Activation functions and derivatives."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def identity(x: np.ndarray) -> np.ndarray:
    return x


def identity_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "identity": (identity, identity_derivative),
}


def get_activation(name: str):
    key = name.lower()
    if key not in ACTIVATIONS:
        supported = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unsupported activation '{name}'. Supported: {supported}")
    return ACTIVATIONS[key]
