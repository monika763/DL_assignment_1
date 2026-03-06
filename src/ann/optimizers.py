"""Parameter optimizers for NumPy models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Optimizer:
    learning_rate: float
    state: dict = field(default_factory=dict)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


@dataclass
class SGD(Optimizer):
    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for key, grad in grads.items():
            params[key] -= self.learning_rate * grad


@dataclass
class Momentum(Optimizer):
    beta: float = 0.9

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for key, grad in grads.items():
            velocity_key = f"v_{key}"
            if velocity_key not in self.state:
                self.state[velocity_key] = np.zeros_like(grad)
            self.state[velocity_key] = self.beta * self.state[velocity_key] + (1.0 - self.beta) * grad
            params[key] -= self.learning_rate * self.state[velocity_key]


@dataclass
class RMSprop(Optimizer):
    beta: float = 0.9
    epsilon: float = 1e-8

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for key, grad in grads.items():
            sq_key = f"s_{key}"
            if sq_key not in self.state:
                self.state[sq_key] = np.zeros_like(grad)
            self.state[sq_key] = self.beta * self.state[sq_key] + (1.0 - self.beta) * (grad * grad)
            params[key] -= self.learning_rate * grad / (np.sqrt(self.state[sq_key]) + self.epsilon)


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key, grad in grads.items():
            m_key = f"m_{key}"
            v_key = f"v_{key}"
            if m_key not in self.state:
                self.state[m_key] = np.zeros_like(grad)
            if v_key not in self.state:
                self.state[v_key] = np.zeros_like(grad)

            self.state[m_key] = self.beta1 * self.state[m_key] + (1.0 - self.beta1) * grad
            self.state[v_key] = self.beta2 * self.state[v_key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.state[m_key] / (1.0 - self.beta1**self.t)
            v_hat = self.state[v_key] / (1.0 - self.beta2**self.t)
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


@dataclass
class NesterovMomentum(Optimizer):
    beta: float = 0.9

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for key, grad in grads.items():
            velocity_key = f"v_{key}"
            if velocity_key not in self.state:
                self.state[velocity_key] = np.zeros_like(grad)

            v_prev = self.state[velocity_key]
            v_new = self.beta * v_prev - self.learning_rate * grad
            # Nesterov lookahead update.
            params[key] += -self.beta * v_prev + (1.0 + self.beta) * v_new
            self.state[velocity_key] = v_new


@dataclass
class Nadam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    t: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key, grad in grads.items():
            m_key = f"m_{key}"
            v_key = f"v_{key}"

            if m_key not in self.state:
                self.state[m_key] = np.zeros_like(grad)
            if v_key not in self.state:
                self.state[v_key] = np.zeros_like(grad)

            self.state[m_key] = self.beta1 * self.state[m_key] + (1.0 - self.beta1) * grad
            self.state[v_key] = self.beta2 * self.state[v_key] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.state[m_key] / (1.0 - self.beta1**self.t)
            v_hat = self.state[v_key] / (1.0 - self.beta2**self.t)

            m_nesterov = self.beta1 * m_hat + ((1.0 - self.beta1) * grad) / (1.0 - self.beta1**self.t)
            params[key] -= self.learning_rate * m_nesterov / (np.sqrt(v_hat) + self.epsilon)


def get_optimizer(name: str, learning_rate: float) -> Optimizer:
    key = name.lower()
    if key == "sgd":
        return SGD(learning_rate=learning_rate)
    if key == "momentum":
        return Momentum(learning_rate=learning_rate)
    if key in {"nesterov", "nesterov_momentum", "nestrov"}:
        return NesterovMomentum(learning_rate=learning_rate)
    if key == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    if key == "adam":
        return Adam(learning_rate=learning_rate)
    if key == "nadam":
        return Nadam(learning_rate=learning_rate)
    supported = "sgd, momentum, nesterov, rmsprop, adam, nadam"
    raise ValueError(f"Unsupported optimizer '{name}'. Supported: {supported}")
