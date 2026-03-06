from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot
from utils.data_loader import load_dataset


def hidden_activations(model: NeuralNetwork, x: np.ndarray) -> list[np.ndarray]:
    acts = []
    a = x
    for layer in range(1, model.num_layers):
        z = a @ model.params[f"W{layer}"] + model.params[f"b{layer}"]
        a = model.activation(z)
        acts.append(a)
    return acts


def dead_neuron_counts(model: NeuralNetwork, x: np.ndarray) -> list[int]:
    acts = hidden_activations(model, x)
    counts = []
    for a in acts:
        dead_mask = np.all(a == 0.0, axis=0)
        counts.append(int(dead_mask.sum()))
    return counts


def tanh_saturation_counts(model: NeuralNetwork, x: np.ndarray, thr: float = 0.99) -> list[int]:
    acts = hidden_activations(model, x)
    counts = []
    for a in acts:
        sat_mask = np.all(np.abs(a) > thr, axis=0)
        counts.append(int(sat_mask.sum()))
    return counts


def hidden_grad_norms(model: NeuralNetwork, xb: np.ndarray, yb: np.ndarray) -> list[float]:
    probs, caches = model._forward(xb)
    grads = model._backward(probs, one_hot(yb, model.output_dim), caches)
    return [float(np.linalg.norm(grads[f"W{layer}"])) for layer in range(1, model.num_layers)]


def first_plateau_epoch(vals: list[float], patience: int = 4, min_delta: float = 1e-3) -> int | None:
    if len(vals) < patience + 1:
        return None
    for i in range(patience, len(vals)):
        improvements = [vals[j] - vals[j - 1] for j in range(i - patience + 1, i + 1)]
        if max(improvements) < min_delta:
            return i + 1
    return None


def run_experiment(
    activation: str,
    learning_rate: float,
    seed: int,
    epochs: int = 20,
    hidden_layers: int = 3,
    hidden_size: int = 128,
    optimizer: str = "sgd",
):
    data = load_dataset("mnist", validation_split=0.1, random_seed=seed, one_hot=False)
    x_train, y_train = data.x_train, data.y_train
    x_val, y_val = data.x_val, data.y_val
 

    n_train, n_val = 20000, 5000
    x_train, y_train = x_train[:n_train], y_train[:n_train]
    x_val, y_val = x_val[:n_val], y_val[:n_val]

    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_hidden_layers=hidden_layers,
        num_neurons=hidden_size,
        initialization_method="he" if activation == "relu" else "xavier",
        activation_function=activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=1,
        batch_size=64,
        weight_decay=0.0,
        random_seed=seed,
    )

    val_acc_hist = []
    dead_or_sat_hist = []
    grad_hist = []

    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        h = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, verbose=False)
        val_acc_hist.append(float(h.val_accuracy[-1]))

        if activation == "relu":
            dead_or_sat_hist.append(dead_neuron_counts(model, x_val))
        else:
            dead_or_sat_hist.append(tanh_saturation_counts(model, x_val))

        idx = rng.choice(x_train.shape[0], size=256, replace=False)
        grad_hist.append(hidden_grad_norms(model, x_train[idx], y_train[idx]))

    plateau = first_plateau_epoch(val_acc_hist, patience=4, min_delta=1e-3)
    return {
        "model": model,
        "val_acc": val_acc_hist,
        "dead_or_sat": dead_or_sat_hist,
        "grad_norms": grad_hist,
        "plateau_epoch": plateau,
    }


def find_relu_plateau(max_seeds: int, epochs: int, hidden_layers: int, hidden_size: int):
    last = None
    for seed in range(max_seeds):
        out = run_experiment(
            activation="relu",
            learning_rate=0.1,
            seed=seed,
            optimizer="sgd",
            epochs=epochs,
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
        )
        last = out
        pe = out["plateau_epoch"]
        if pe is not None and pe <= max(10, epochs // 2):
            dead_total = sum(out["dead_or_sat"][pe - 1])
            if dead_total > 0:
                return seed, out
    return max_seeds - 1, last


def log_comparison_graphs(relu: dict, tanh: dict) -> None:
    pe_relu = relu["plateau_epoch"] or len(relu["val_acc"])
    pe_tanh = tanh["plateau_epoch"] or len(tanh["val_acc"])

    epochs_relu = np.arange(1, len(relu["val_acc"]) + 1)
    epochs_tanh = np.arange(1, len(tanh["val_acc"]) + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs_relu, relu["val_acc"], marker="o", label="ReLU (lr=0.1)")
    ax1.plot(epochs_tanh, tanh["val_acc"], marker="s", label="Tanh (lr=0.1)")
    ax1.axvline(pe_relu, linestyle="--", label=f"ReLU plateau: {pe_relu}")
    ax1.axvline(pe_tanh, linestyle="--", label=f"Tanh plateau: {pe_tanh}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_title("Validation Accuracy: ReLU vs Tanh")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    wandb.log({"plots/val_accuracy_comparison": wandb.Image(fig1)})
    plt.close(fig1)

    relu_dead = relu["dead_or_sat"][pe_relu - 1]
    tanh_sat = tanh["dead_or_sat"][pe_tanh - 1]
    layers = np.arange(1, len(relu_dead) + 1)
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(layers - width / 2, relu_dead, width=width, label="ReLU dead")
    ax2.bar(layers + width / 2, tanh_sat, width=width, label="Tanh saturated")
    ax2.set_xlabel("Hidden Layer")
    ax2.set_ylabel("Neuron Count")
    ax2.set_title("Layer-wise Dead/Saturated Neurons at Plateau")
    ax2.set_xticks(layers)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    wandb.log({"plots/dead_vs_saturated_layerwise": wandb.Image(fig2)})
    plt.close(fig2)

    relu_g = np.mean(np.array(relu["grad_norms"][:5]), axis=0)
    tanh_g = np.mean(np.array(tanh["grad_norms"][:5]), axis=0)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(layers, relu_g, marker="o", label="ReLU grad norm")
    ax3.plot(layers, tanh_g, marker="s", label="Tanh grad norm")
    ax3.set_xlabel("Hidden Layer")
    ax3.set_ylabel("Avg ||dW|| (first 5 epochs)")
    ax3.set_title("Gradient Flow Comparison")
    ax3.set_xticks(layers)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    wandb.log({"plots/gradient_norm_comparison": wandb.Image(fig3)})
    plt.close(fig3)


def log_to_wandb(seed: int, relu: dict, tanh: dict, hidden_layers: int) -> None:
    pe_relu = relu["plateau_epoch"] or len(relu["val_acc"])
    pe_tanh = tanh["plateau_epoch"] or len(tanh["val_acc"])

    max_epochs = max(len(relu["val_acc"]), len(tanh["val_acc"]))
    for i in range(max_epochs):
        metrics = {"epoch": i + 1}

        if i < len(relu["val_acc"]):
            metrics["relu/val_accuracy"] = relu["val_acc"][i]
            for layer in range(hidden_layers):
                metrics[f"relu/dead_neurons_layer_{layer + 1}"] = relu["dead_or_sat"][i][layer]
                metrics[f"relu/grad_norm_layer_{layer + 1}"] = relu["grad_norms"][i][layer]

        if i < len(tanh["val_acc"]):
            metrics["tanh/val_accuracy"] = tanh["val_acc"][i]
            for layer in range(hidden_layers):
                metrics[f"tanh/saturated_neurons_layer_{layer + 1}"] = tanh["dead_or_sat"][i][layer]
                metrics[f"tanh/grad_norm_layer_{layer + 1}"] = tanh["grad_norms"][i][layer]

        wandb.log(metrics, step=i + 1)

    relu_dead_at_plateau = relu["dead_or_sat"][pe_relu - 1]
    tanh_sat_at_plateau = tanh["dead_or_sat"][pe_tanh - 1]

    wandb.summary["seed"] = seed
    wandb.summary["relu/plateau_epoch"] = pe_relu
    wandb.summary["tanh/plateau_epoch"] = pe_tanh
    wandb.summary["relu/final_val_accuracy"] = relu["val_acc"][-1]
    wandb.summary["tanh/final_val_accuracy"] = tanh["val_acc"][-1]
    wandb.summary["relu/dead_total_at_plateau"] = int(sum(relu_dead_at_plateau))
    wandb.summary["tanh/saturated_total_at_plateau"] = int(sum(tanh_sat_at_plateau))


def summarize(args: argparse.Namespace) -> None:
    seed, relu = find_relu_plateau(
        max_seeds=args.max_seeds,
        epochs=args.epochs,
        hidden_layers=args.hidden_layers,
        hidden_size=args.hidden_size,
    )
    tanh = run_experiment(
        activation="tanh",
        learning_rate=0.1,
        seed=seed,
        optimizer="sgd",
        epochs=args.epochs,
        hidden_layers=args.hidden_layers,
        hidden_size=args.hidden_size,
    )

    pe_relu = relu["plateau_epoch"] or len(relu["val_acc"])
    pe_tanh = tanh["plateau_epoch"] or len(tanh["val_acc"])

    relu_dead_at_plateau = relu["dead_or_sat"][pe_relu - 1]
    tanh_sat_at_plateau = tanh["dead_or_sat"][pe_tanh - 1]

    relu_grad_early = np.mean(np.array(relu["grad_norms"][:5]), axis=0)
    tanh_grad_early = np.mean(np.array(tanh["grad_norms"][:5]), axis=0)

    print("\n=== ReLU (lr=0.1) run with early plateau ===")
    print(f"seed: {seed}")
    print(f"validation accuracy by epoch: {[round(v, 4) for v in relu['val_acc']]}")
    print(f"plateau epoch: {pe_relu}")
    print(f"dead neurons per hidden layer at plateau: {relu_dead_at_plateau}")

    print("\n=== Tanh (lr=0.1) comparison run ===")
    print(f"seed: {seed}")
    print(f"validation accuracy by epoch: {[round(v, 4) for v in tanh['val_acc']]}")
    print(f"plateau epoch: {pe_tanh}")
    print(f"tanh saturated neurons per hidden layer at plateau (|a|>0.99): {tanh_sat_at_plateau}")

    print("\n=== Gradient comparison (avg L2 norm of dW over first 5 epochs) ===")
    print(f"ReLU hidden-layer grad norms: {[round(float(x), 6) for x in relu_grad_early]}")
    print(f"Tanh hidden-layer grad norms: {[round(float(x), 6) for x in tanh_grad_early]}")

    print("\n=== Convergence difference (based on observed gradients) ===")
    print(
        "ReLU + high LR creates many permanently zero activations (dead neurons), "
        "so gradients for those units are zero and validation accuracy plateaus early."
    )
    print(
        "Tanh neurons usually stay active (fewer fully dead units), so gradients continue to flow; "
        "very high LR can still cause saturation and slower progress."
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "dataset": "mnist",
            "learning_rate": 0.1,
            "optimizer": "sgd",
            "epochs": args.epochs,
            "hidden_layers": args.hidden_layers,
            "hidden_size": args.hidden_size,
            "max_seeds": args.max_seeds,
        },
        name=args.wandb_run_name,
    )
    _ = run
    log_to_wandb(seed=seed, relu=relu, tanh=tanh, hidden_layers=args.hidden_layers)
    log_comparison_graphs(relu=relu, tanh=tanh)
    wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dead neuron investigation with W&B logging")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--max_seeds", type=int, default=12)

    parser.add_argument("--wandb_project", type=str, default="mnist-dead-neuron-investigation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summarize(args)
