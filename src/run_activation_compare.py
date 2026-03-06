"""Compare ReLU vs Sigmoid with Adam and track first-layer gradient norms."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot
from train import load_dataset, train_val_split


@dataclass(frozen=True)
class NetConfig:
    hidden_layers: int
    neurons: int

    def label(self) -> str:
        return f"L{self.hidden_layers}_N{self.neurons}"


def parse_configs(configs_text: str) -> list[NetConfig]:
    configs: list[NetConfig] = []
    for token in configs_text.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if "x" not in value:
            raise ValueError(f"Invalid config '{token}'. Expected format like '3x128'.")
        layers_text, neurons_text = value.split("x", maxsplit=1)
        configs.append(NetConfig(hidden_layers=int(layers_text), neurons=int(neurons_text)))

    if not configs:
        raise ValueError("Provide at least one network config.")
    return configs


def train_with_grad_norms(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: NetConfig,
    activation: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    initialization_method: str,
    random_seed: int,
) -> dict[str, list[float]]:
    np.random.seed(random_seed)

    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=config.neurons,
        num_hidden_layers=config.hidden_layers,
        initialization_method=initialization_method,
        activation_function=activation,
        optimizer="adam",
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "grad_norm_w1": [],
    }

    y_train_int = y_train.astype(int).ravel()
    y_val_int = y_val.astype(int).ravel()
    train_targets = one_hot(y_train_int, model.output_dim)
    val_targets = one_hot(y_val_int, model.output_dim)

    for epoch in range(1, epochs + 1):
        batch_grad_norms: list[float] = []
        for xb, yb in model._iter_minibatches(x_train, y_train_int, batch_size):
            yb_onehot = one_hot(yb, model.output_dim)
            probs, caches = model._forward(xb)
            grads = model._backward(probs, yb_onehot, caches)
            batch_grad_norms.append(float(np.linalg.norm(grads["W1"])))
            model.optimizer.step(model.params, grads)

        train_probs, _ = model._forward(x_train)
        val_probs, _ = model._forward(x_val)

        train_loss = float(model._compute_loss(train_probs, train_targets))
        val_loss = float(model._compute_loss(val_probs, val_targets))
        grad_norm = float(np.mean(batch_grad_norms))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["grad_norm_w1"].append(grad_norm)

        key_prefix = f"{config.label()}/{activation}"
        wandb.log(
            {
                f"{key_prefix}/train_loss": train_loss,
                f"{key_prefix}/val_loss": val_loss,
                f"{key_prefix}/grad_norm_w1": grad_norm,
            },
            step=epoch,
        )

    return history


def make_plot(
    all_results: dict[tuple[str, str], dict[str, list[float]]],
    configs: list[NetConfig],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(configs), 1, figsize=(9, 3.5 * len(configs)), sharex=True)
    if len(configs) == 1:
        axes = [axes]

    for axis, config in zip(axes, configs):
        relu_curve = all_results[(config.label(), "relu")]["grad_norm_w1"]
        sigmoid_curve = all_results[(config.label(), "sigmoid")]["grad_norm_w1"]
        epochs = np.arange(1, len(relu_curve) + 1)

        axis.plot(epochs, relu_curve, marker="o", label="ReLU")
        axis.plot(epochs, sigmoid_curve, marker="o", label="Sigmoid")
        axis.set_yscale("log")
        axis.set_ylabel("||dL/dW1||")
        axis.set_title(f"{config.label()} | Adam")
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("First Hidden Layer Gradient Norms (Adam): ReLU vs Sigmoid", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def detect_vanishing(
    all_results: dict[tuple[str, str], dict[str, list[float]]],
    configs: list[NetConfig],
) -> tuple[bool, list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    signal_count = 0

    for config in configs:
        relu = all_results[(config.label(), "relu")]["grad_norm_w1"]
        sigmoid = all_results[(config.label(), "sigmoid")]["grad_norm_w1"]
        relu_final = relu[-1]
        sigmoid_final = sigmoid[-1]
        ratio = sigmoid_final / (relu_final + 1e-12)
        has_signal = ratio < 0.1
        signal_count += int(has_signal)

        rows.append(
            {
                "config": config.label(),
                "relu_start": relu[0],
                "relu_final": relu_final,
                "sigmoid_start": sigmoid[0],
                "sigmoid_final": sigmoid_final,
                "sigmoid_to_relu_final_ratio": ratio,
                "vanishing_signal": has_signal,
            }
        )

    observed = signal_count > 0
    if observed:
        note = "Yes. Sigmoid shows much smaller first-layer gradient norms than ReLU (vanishing-gradient behavior)."
    else:
        note = "No clear vanishing-gradient signal in this run."
    return observed, rows, note


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fix optimizer=Adam and compare ReLU vs Sigmoid across network configurations."
    )
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--configs", type=str, default="2x128,3x128,5x128")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--initialization_method", type=str, default="xavier")
    parser.add_argument("--plot_path", type=str, default="activation_grad_norms.png")

    parser.add_argument("--wandb_project", type=str, default="mnist-ann")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_name", type=str, default="adam-sigmoid-vs-relu-gradnorm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configs = parse_configs(args.configs)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
    )
    wandb.config.update(
        {
            "dataset": args.dataset,
            "optimizer": "adam",
            "activations": ["relu", "sigmoid"],
            "configs": [cfg.label() for cfg in configs],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "random_seed": args.random_seed,
            "initialization_method": args.initialization_method,
        },
        allow_val_change=True,
    )

    np.random.seed(args.random_seed)
    (x_train_full, y_train_full), _ = load_dataset(args.dataset)
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full, y_train_full, val_ratio=args.val_ratio, random_seed=args.random_seed
    )

    all_results: dict[tuple[str, str], dict[str, list[float]]] = {}
    for config in configs:
        for activation in ("relu", "sigmoid"):
            print(f"Training config={config.label()}, activation={activation}, optimizer=adam")
            all_results[(config.label(), activation)] = train_with_grad_norms(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=config,
                activation=activation,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                initialization_method=args.initialization_method,
                random_seed=args.random_seed,
            )

    plot_path = (PROJECT_ROOT / args.plot_path).resolve()
    make_plot(all_results, configs, plot_path)
    wandb.log({"plots/grad_norm_w1_relu_vs_sigmoid": wandb.Image(str(plot_path))})

    observed, rows, note = detect_vanishing(all_results, configs)
    table = wandb.Table(
        columns=[
            "config",
            "relu_start",
            "relu_final",
            "sigmoid_start",
            "sigmoid_final",
            "sigmoid_to_relu_final_ratio",
            "vanishing_signal",
        ],
        data=[
            [
                row["config"],
                row["relu_start"],
                row["relu_final"],
                row["sigmoid_start"],
                row["sigmoid_final"],
                row["sigmoid_to_relu_final_ratio"],
                row["vanishing_signal"],
            ]
            for row in rows
        ],
    )
    wandb.log({"vanishing_gradient_summary": table})

    wandb.summary["vanishing_gradient_observed"] = observed
    wandb.summary["observation_note"] = note

    print(f"W&B run: {run.url}")
    print(f"Plot saved: {plot_path}")
    print(f"Observation: {note}")
    wandb.finish()


if __name__ == "__main__":
    main()
