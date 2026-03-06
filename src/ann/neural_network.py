"""
Fully connected neural network implemented with NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .activations import get_activation
from .objective_functions import accuracy_from_probs, cross_entropy_loss, one_hot, softmax
from .optimizers import Optimizer, get_optimizer


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


class NeuralNetwork:

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: int | Iterable[int] = 128,
        num_hidden_layers: int = 2,
        initialization_method: str = "xavier",
        activation_function: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        loss_function: str = "cross_entropy",
        random_seed: int | None = None,
    ):

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.initialization_method = initialization_method.lower()
        self.activation_function = activation_function.lower()
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.loss_function = loss_function.lower()
        self.random_seed = random_seed

        self.hidden_layer_sizes = self._resolve_hidden_sizes(num_neurons, self.num_hidden_layers)

        self.layer_dims = [self.input_dim, *self.hidden_layer_sizes, self.output_dim]

        self.num_layers = len(self.layer_dims) - 1

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.activation, self.activation_derivative = get_activation(self.activation_function)

        self.params = self._initialize_parameters()

        self.optimizer: Optimizer = get_optimizer(optimizer, self.learning_rate)

        # Debug statistics for experiments
        self.debug_stats = {
            "grad_norm_layer1": [],
            "dead_neuron_ratio": [],
            "neuron_gradients": []
        }

    @staticmethod
    def _resolve_hidden_sizes(num_neurons: int | Iterable[int], num_hidden_layers: int) -> list[int]:

        if isinstance(num_neurons, int):
            return [int(num_neurons)] * num_hidden_layers

        sizes = [int(x) for x in num_neurons]

        if len(sizes) != num_hidden_layers:
            raise ValueError("num_neurons length must match num_hidden_layers")

        return sizes

    def _initialize_parameters(self):

        params = {}

        for layer in range(1, self.num_layers + 1):

            fan_in = self.layer_dims[layer - 1]
            fan_out = self.layer_dims[layer]

            params[f"b{layer}"] = np.zeros((1, fan_out))

            if self.initialization_method == "xavier":
                std = np.sqrt(2.0 / (fan_in + fan_out))
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * std

            elif self.initialization_method == "he":
                std = np.sqrt(2.0 / fan_in)
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * std

            elif self.initialization_method == "normal":
                params[f"W{layer}"] = np.random.randn(fan_in, fan_out) * 0.01

            elif self.initialization_method == "zeros":
                params[f"W{layer}"] = np.zeros((fan_in, fan_out))

            else:
                raise ValueError("Unsupported initialization")

        return params

    def _forward(self, x):

        caches = []

        activation_prev = x

        for layer in range(1, self.num_layers):

            weights = self.params[f"W{layer}"]
            bias = self.params[f"b{layer}"]

            z = activation_prev @ weights + bias

            activation_next = self.activation(z)

            # Dead neuron monitoring
            zero_ratio = np.mean(activation_next == 0)

            self.debug_stats["dead_neuron_ratio"].append(float(zero_ratio))

            caches.append((activation_prev, z, weights, bias))

            activation_prev = activation_next

        weights_out = self.params[f"W{self.num_layers}"]
        bias_out = self.params[f"b{self.num_layers}"]

        logits = activation_prev @ weights_out + bias_out

        probs = softmax(logits)

        caches.append((activation_prev, logits, weights_out, bias_out))

        return probs, caches

    def _compute_loss(self, probs, targets):

        if self.loss_function == "cross_entropy":

            base_loss = cross_entropy_loss(probs, targets)

        elif self.loss_function == "mse":

            base_loss = np.mean((probs - targets) ** 2)

        else:

            raise ValueError("Unsupported loss function")

        if self.weight_decay <= 0.0:
            return base_loss

        l2 = 0.0

        for layer in range(1, self.num_layers + 1):
            l2 += np.sum(self.params[f"W{layer}"] ** 2)

        return base_loss + 0.5 * self.weight_decay * l2 / targets.shape[0]

    def _backward(self, probs, targets, caches):

        grads = {}

        batch_size = targets.shape[0]

        dlogits = (probs - targets) / batch_size

        prev_activation, _, weights, _ = caches[-1]

        grads[f"W{self.num_layers}"] = prev_activation.T @ dlogits

        grads[f"b{self.num_layers}"] = np.sum(dlogits, axis=0, keepdims=True)

        dactivation = dlogits @ weights.T

        for layer in range(self.num_layers - 1, 0, -1):

            prev_activation, z, weights, _ = caches[layer - 1]

            dz = dactivation * self.activation_derivative(z)

            grads[f"W{layer}"] = prev_activation.T @ dz

            grads[f"b{layer}"] = np.sum(dz, axis=0, keepdims=True)

            dactivation = dz @ weights.T

        # Gradient norm logging
        grad_norm = np.linalg.norm(grads["W1"])

        self.debug_stats["grad_norm_layer1"].append(float(grad_norm))

        # Track gradients of 5 neurons
        neuron_grads = grads["W1"][:5].mean(axis=0)

        self.debug_stats["neuron_gradients"].append(neuron_grads.tolist())

        return grads

    def _iter_minibatches(self, x, y):

        indices = np.random.permutation(x.shape[0])

        for start in range(0, x.shape[0], self.batch_size):

            batch_idx = indices[start:start + self.batch_size]

            yield x[batch_idx], y[batch_idx]

    def fit(self, x_train, y_train, x_val=None, y_val=None, verbose=True):

        history = TrainingHistory()

        train_targets = one_hot(y_train, self.output_dim)

        for epoch in range(self.epochs):

            for xb, yb in self._iter_minibatches(x_train, y_train):

                yb_onehot = one_hot(yb, self.output_dim)

                probs, caches = self._forward(xb)

                grads = self._backward(probs, yb_onehot, caches)

                self.optimizer.step(self.params, grads)

            train_probs, _ = self._forward(x_train)

            train_loss = self._compute_loss(train_probs, train_targets)

            train_acc = accuracy_from_probs(train_probs, y_train)

            history.train_loss.append(train_loss)

            history.train_accuracy.append(train_acc)

            if x_val is not None:

                val_targets = one_hot(y_val, self.output_dim)

                val_probs, _ = self._forward(x_val)

                val_loss = self._compute_loss(val_probs, val_targets)

                val_acc = accuracy_from_probs(val_probs, y_val)

                history.val_loss.append(val_loss)

                history.val_accuracy.append(val_acc)

            if verbose:

                print(
                    f"Epoch {epoch+1}/{self.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f}"
                )

        return history

    def predict_proba(self, x):

        probs, _ = self._forward(x)

        return probs

    def predict(self, x):

        return np.argmax(self.predict_proba(x), axis=1)

    def evaluate(self, x, y):

        targets = one_hot(y, self.output_dim)

        probs = self.predict_proba(x)

        loss = self._compute_loss(probs, targets)

        acc = accuracy_from_probs(probs, y)

        return loss, acc