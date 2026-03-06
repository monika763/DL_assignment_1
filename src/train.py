"""Train the NumPy ANN on MNIST with W&B logging and sweep support."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import wandb
import yaml
from keras.datasets import fashion_mnist, mnist


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ann.neural_network import NeuralNetwork


def parse_neuron_config(value: int | str | list[int], num_hidden_layers: int) -> int | list[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        if len(value) == 1:
            return int(value[0])
        if len(value) != num_hidden_layers:
            raise ValueError(
                f"When passing list --num_neurons, expected {num_hidden_layers} values, got {len(value)}."
            )
        return [int(x) for x in value]

    text_value = str(value).strip()
    if "," not in text_value:
        return int(text_value)

    layer_sizes = [int(x.strip()) for x in text_value.split(",") if x.strip()]
    if len(layer_sizes) == 1:
        return layer_sizes[0]
    if len(layer_sizes) != num_hidden_layers:
        raise ValueError(
            f"When passing comma-separated --num_neurons, expected {num_hidden_layers} values, got {len(layer_sizes)}."
        )
    return layer_sizes


def train_val_split(
    x: np.ndarray, y: np.ndarray, val_ratio: float, random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)

    val_count = int(x.shape[0] * val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def load_dataset(name: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    dataset_name = name.lower()
    if dataset_name == "mnist":
        return mnist.load_data()
    if dataset_name == "fashion_mnist":
        return fashion_mnist.load_data()
    raise ValueError("Unsupported dataset. Use 'mnist' or 'fashion_mnist'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a configurable NumPy ANN on MNIST.")

    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument(
        "--num_neurons",
        type=str,
        default="128",
        help="Single integer for all hidden layers or comma-separated list per layer, e.g. 128,64,32",
    )
    parser.add_argument("--initialization_method", type=str, default="xavier")
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nesterov", "nestrov", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--wandb_project", type=str, default="mnist-ann")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_sweep", action="store_true", help="Launch a W&B sweep using sweep_config.")
    parser.add_argument(
        "--sweep_config",
        type=str,
        default=str(PROJECT_ROOT / "sweep_config.yaml"),
        help="Path to W&B sweep YAML config.",
    )
    parser.add_argument("--sweep_count", type=int, default=100, help="Number of sweep runs.")

    return parser


def get_base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "num_hidden_layers": args.num_hidden_layers,
        "num_neurons": args.num_neurons,
        "initialization_method": args.initialization_method,
        "activation_function": args.activation_function,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "random_seed": args.random_seed,
    }


def _first_epoch_at_or_above(values: list[float], target: float) -> int | None:
    for idx, value in enumerate(values, start=1):
        if value >= target:
            return idx
    return None


def compare_optimizer_convergence(
    dataset: str = "mnist",
    optimizers: tuple[str, ...] = ("sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"),
    num_hidden_layers: int = 3,
    num_neurons: int | list[int] = 128,
    activation_function: str = "relu",
    initialization_method: str = "he",
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    target_val_accuracy: float = 0.90,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Compare optimizer convergence using the same architecture and data split.

    The default configuration uses 3 hidden layers with 128 neurons each and
    ReLU activations, matching the assignment requirement.
    """
    optimizer_list = tuple(opt.lower() for opt in optimizers)
    if not optimizer_list:
        raise ValueError("optimizers cannot be empty.")

    np.random.seed(random_seed)

    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(dataset)
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    y_test = y_test.astype(int)

    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full,
        y_train_full,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    summary: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}

    for optimizer_name in optimizer_list:
        model = NeuralNetwork(
            input_dim=x_train.shape[1],
            output_dim=10,
            num_neurons=num_neurons,
            num_hidden_layers=num_hidden_layers,
            initialization_method=initialization_method,
            activation_function=activation_function,
            optimizer=optimizer_name,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            random_seed=random_seed,
        )

        history = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, verbose=verbose)
        histories[optimizer_name] = history

        tracked_loss = history.val_loss if history.val_loss else history.train_loss
        tracked_acc = history.val_accuracy if history.val_accuracy else history.train_accuracy

        loss_drop = float(tracked_loss[0] - tracked_loss[-1]) if tracked_loss else 0.0
        avg_loss_drop_per_epoch = loss_drop / max(1, len(tracked_loss) - 1)
        epochs_to_target = _first_epoch_at_or_above(tracked_acc, target_val_accuracy)
        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        summary.append(
            {
                "optimizer": optimizer_name,
                "final_train_loss": history.train_loss[-1],
                "final_train_accuracy": history.train_accuracy[-1],
                "final_val_loss": history.val_loss[-1] if history.val_loss else None,
                "final_val_accuracy": history.val_accuracy[-1] if history.val_accuracy else None,
                "avg_loss_drop_per_epoch": avg_loss_drop_per_epoch,
                "epochs_to_target_val_accuracy": epochs_to_target,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

    summary.sort(
        key=lambda row: (
            row["epochs_to_target_val_accuracy"] is None,
            row["epochs_to_target_val_accuracy"] if row["epochs_to_target_val_accuracy"] is not None else float("inf"),
            -row["avg_loss_drop_per_epoch"],
        )
    )

    return {
        "config": {
            "dataset": dataset,
            "num_hidden_layers": num_hidden_layers,
            "num_neurons": num_neurons,
            "activation_function": activation_function,
            "initialization_method": initialization_method,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "val_ratio": val_ratio,
            "random_seed": random_seed,
            "target_val_accuracy": target_val_accuracy,
        },
        "summary": summary,
        "histories": histories,
    }


def train_single_run(args: argparse.Namespace, sweep_run: bool = False) -> None:
    base_config = get_base_config(args)

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
    }
    if not sweep_run:
        init_kwargs["name"] = args.wandb_run_name
        init_kwargs["config"] = base_config

    run = wandb.init(**init_kwargs)
    cfg = {**base_config, **dict(run.config)}
    wandb.config.update(cfg, allow_val_change=True)

    np.random.seed(int(cfg["random_seed"]))

    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(str(cfg["dataset"]))
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    y_test = y_test.astype(int)

    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full,
        y_train_full,
        float(cfg["val_ratio"]),
        random_seed=int(cfg["random_seed"]),
    )

    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_neurons = parse_neuron_config(cfg["num_neurons"], num_hidden_layers)
    wandb.config.update(
        {
            "input_dim": x_train.shape[1],
            "output_dim": 10,
            "num_neurons_resolved": str(num_neurons),
        },
        allow_val_change=True,
    )

    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=num_neurons,
        num_hidden_layers=num_hidden_layers,
        initialization_method=str(cfg["initialization_method"]),
        activation_function=str(cfg["activation_function"]),
        optimizer=str(cfg["optimizer"]),
        learning_rate=float(cfg["learning_rate"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        weight_decay=float(cfg["weight_decay"]),
        random_seed=int(cfg["random_seed"]),
    )

    history = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, verbose=True)

    for epoch_idx in range(len(history.train_loss)):
        metrics = {
            "epoch": epoch_idx + 1,
            "train/loss": history.train_loss[epoch_idx],
            "train/accuracy": history.train_accuracy[epoch_idx],
        }
        if epoch_idx < len(history.val_loss):
            metrics["val/loss"] = history.val_loss[epoch_idx]
            metrics["val/accuracy"] = history.val_accuracy[epoch_idx]
        wandb.log(metrics, step=epoch_idx + 1)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    wandb.log({"test/loss": test_loss, "test/accuracy": test_accuracy})
    wandb.summary["best_val_accuracy"] = max(history.val_accuracy) if history.val_accuracy else None
    wandb.summary["test_loss"] = test_loss
    wandb.summary["test_accuracy"] = test_accuracy

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    wandb.finish()


def run_sweep(args: argparse.Namespace) -> None:
    with open(args.sweep_config, "r", encoding="utf-8") as fp:
        sweep_config = yaml.safe_load(fp)

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    print(f"Created sweep: {sweep_id}")
    wandb.agent(sweep_id, function=lambda: train_single_run(args, sweep_run=True), count=args.sweep_count)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.run_sweep:
        run_sweep(args)
    else:
        train_single_run(args, sweep_run=False)


if __name__ == "__main__":
    main()
