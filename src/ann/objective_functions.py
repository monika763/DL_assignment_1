"""Objective functions and metrics."""

from __future__ import annotations

import numpy as np


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int).ravel()
    encoded = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))


def accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y_true))
