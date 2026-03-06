from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot
from utils.data_loader import load_dataset


def eval_cross_entropy(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    probs, _ = model._forward(x)
    targets = one_hot(y, model.output_dim)
    eps = 1e-12
    loss = -np.mean(np.sum(targets * np.log(np.clip(probs, eps, 1.0 - eps)), axis=1))
    acc = float(np.mean(np.argmax(probs, axis=1) == y))
    return float(loss), acc


def eval_mse(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    probs, _ = model._forward(x)
    targets = one_hot(y, model.output_dim)
    loss = float(np.mean((probs - targets) ** 2))
    acc = float(np.mean(np.argmax(probs, axis=1) == y))
    return loss, acc


def backward_from_dlogits(model: NeuralNetwork, dlogits: np.ndarray, caches: list[Any]) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    batch_size = dlogits.shape[0]

    a_prev, _, w_last, _ = caches[-1]
    grads[f"W{model.num_layers}"] = a_prev.T @ dlogits + (model.weight_decay / batch_size) * w_last
    grads[f"b{model.num_layers}"] = np.sum(dlogits, axis=0, keepdims=True)

    da = dlogits @ w_last.T
    for layer in range(model.num_layers - 1, 0, -1):
        a_prev, z, w, _ = caches[layer - 1]
        dz = da * model.activation_derivative(z)
        grads[f"W{layer}"] = a_prev.T @ dz + (model.weight_decay / batch_size) * w
        grads[f"b{layer}"] = np.sum(dz, axis=0, keepdims=True)
        da = dz @ w.T

    return grads


def train_one_epoch_ce(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> None:
    for xb, yb in model._iter_minibatches(x, y, model.batch_size):
        probs, caches = model._forward(xb)
        grads = model._backward(probs, one_hot(yb, model.output_dim), caches)
        model.optimizer.step(model.params, grads)


def train_one_epoch_mse(model: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> None:
    for xb, yb in model._iter_minibatches(x, y, model.batch_size):
        probs, caches = model._forward(xb)
        targets = one_hot(yb, model.output_dim)

        dL_dp = (2.0 / model.output_dim) * (probs - targets)
        dot = np.sum(dL_dp * probs, axis=1, keepdims=True)
        dlogits = probs * (dL_dp - dot)
        dlogits /= xb.shape[0]

        grads = backward_from_dlogits(model, dlogits, caches)
        model.optimizer.step(model.params, grads)


def build_sweep_config(base_seed: int) -> dict[str, Any]:
    return {
        "name": "loss_comparison",
        "method": "grid",
        "metric": {"name": "val/accuracy", "goal": "maximize"},
        "parameters": {
            "loss_function": {"values": ["cross_entropy_loss", "mse_loss"]},
            "seed": {"values": [base_seed + i for i in range(5)]},
        },
    }


def train_run(args: argparse.Namespace) -> None:
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode)
    cfg = wandb.config

    loss_function = str(cfg.loss_function)
    seed = int(cfg.seed)
    run.name = f"{loss_function}seed{seed}"

    np.random.seed(seed)
    data = load_dataset(args.dataset, validation_split=args.val_size, random_seed=seed, one_hot=False)
    x_train, y_train = data.x_train, data.y_train
    x_val, y_val = data.x_val, data.y_val


    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=args.hidden_size,
        num_hidden_layers=args.hidden_layers,
        initialization_method=args.initialization,
        activation_function=args.activation,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        epochs=1,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        random_seed=seed,
    )

    for epoch in range(1, args.epochs + 1):
        if loss_function == "cross_entropy_loss":
            train_one_epoch_ce(model, x_train, y_train)
            train_loss, train_acc = eval_cross_entropy(model, x_train, y_train)
            val_loss, val_acc = eval_cross_entropy(model, x_val, y_val)
        else:
            train_one_epoch_mse(model, x_train, y_train)
            train_loss, train_acc = eval_mse(model, x_train, y_train)
            val_loss, val_acc = eval_mse(model, x_val, y_val)

        wandb.log(
            {
                "epoch": epoch,
                "loss_function": loss_function,
                "seed": seed,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            },
            step=epoch,
        )

    wandb.summary["final/train_loss"] = train_loss
    wandb.summary["final/val_loss"] = val_loss
    wandb.summary["final/val_accuracy"] = val_acc
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep: 5 CE runs + 5 MSE runs with W&B logging")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--initialization", type=str, default="xavier")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument("--wandb_project", type=str, default="loss-function-comparison")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_count", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sweep_config = build_sweep_config(base_seed=args.base_seed)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    print(f"Created sweep: {sweep_id}")
    wandb.agent(sweep_id, function=lambda: train_run(args), count=args.run_count)


if __name__ == "__main__":
    main()