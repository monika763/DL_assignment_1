from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from ann.objective_functions import one_hot
from train import load_dataset, train_val_split


def load_flattened_dataset(name: str):
    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(name)
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    y_test = y_test.astype(int)
    return x_train_full, y_train_full, x_test, y_test


def collect_gradients(
    x_train: np.ndarray,
    y_train: np.ndarray,
    initialization_method: str,
    activation_function: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    iterations: int,
    num_hidden_layers: int,
    num_neurons: int,
    random_seed: int,
):
    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=num_neurons,
        num_hidden_layers=num_hidden_layers,
        initialization_method=initialization_method,
        activation_function=activation_function,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=1,
        batch_size=batch_size,
        weight_decay=0.0,
        random_seed=random_seed,
    )

    grad_tracks = [[] for _ in range(5)]
    loss_track = []

    step = 0
    while step < iterations:
        for xb, yb in model._iter_minibatches(x_train, y_train):
            probs, caches = model._forward(xb)
            targets = one_hot(yb, model.output_dim)
            loss = model._compute_loss(probs, targets)
            grads = model._backward(probs, targets, caches)

            # Per-neuron gradient magnitude in first hidden layer weights.
            per_neuron_grad = np.mean(np.abs(grads["W1"]), axis=0)
            for n in range(5):
                grad_tracks[n].append(float(per_neuron_grad[n]))

            loss_track.append(float(loss))
            model.optimizer.step(model.params, grads)

            step += 1
            if step >= iterations:
                break

    return grad_tracks, loss_track


def plot_gradients(zeros_grads, xavier_grads, out_path: Path):
    iterations = len(zeros_grads[0])
    x = np.arange(1, iterations + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i in range(5):
        axes[0].plot(x, zeros_grads[i], label=f"Neuron {i+1}")
        axes[1].plot(x, xavier_grads[i], label=f"Neuron {i+1}")

    axes[0].set_title("Zeros Initialization")
    axes[1].set_title("Xavier Initialization")
    axes[0].set_xlabel("Training Iteration")
    axes[1].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Mean |Gradient| of Neuron Weights")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Q2.9 Gradient Symmetry: 5 Neurons in Same Hidden Layer", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_loss(zeros_loss, xavier_loss, out_path: Path):
    iterations = len(zeros_loss)
    x = np.arange(1, iterations + 1)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, zeros_loss, label="Zeros loss")
    plt.plot(x, xavier_loss, label="Xavier loss")
    plt.xlabel("Training Iteration")
    plt.ylabel("Mini-batch Loss")
    plt.title("Loss Over First 50 Iterations")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2.9: Weight Initialization & Symmetry")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--activation_function", type=str, default="sigmoid")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_neurons", type=int, default=128)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="src/models")
    parser.add_argument("--wandb_project", type=str, default="mnist-ann-analysis")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_run_name", type=str, default="q2_9_weight_init_symmetry")
    args = parser.parse_args()

    if args.num_neurons < 5:
        raise ValueError("--num_neurons must be >= 5 for plotting 5 neurons.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train_full, _, _ = load_flattened_dataset(args.dataset)
    x_train, y_train, _, _ = train_val_split(
        x_train_full, y_train_full, val_ratio=args.val_ratio, random_seed=args.random_seed
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        config=vars(args),
    )

    zeros_grads, zeros_loss = collect_gradients(
        x_train=x_train,
        y_train=y_train,
        initialization_method="zeros",
        activation_function=args.activation_function,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        iterations=args.iterations,
        num_hidden_layers=args.num_hidden_layers,
        num_neurons=args.num_neurons,
        random_seed=args.random_seed,
    )

    xavier_grads, xavier_loss = collect_gradients(
        x_train=x_train,
        y_train=y_train,
        initialization_method="xavier",
        activation_function=args.activation_function,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        iterations=args.iterations,
        num_hidden_layers=args.num_hidden_layers,
        num_neurons=args.num_neurons,
        random_seed=args.random_seed,
    )

    for i in range(args.iterations):
        log_item = {
            "iteration": i + 1,
            "zeros/loss": zeros_loss[i],
            "xavier/loss": xavier_loss[i],
        }
        for n in range(5):
            log_item[f"zeros/grad_neuron_{n+1}"] = zeros_grads[n][i]
            log_item[f"xavier/grad_neuron_{n+1}"] = xavier_grads[n][i]
        wandb.log(log_item, step=i + 1)

    zeros_stack = np.array(zeros_grads, dtype=np.float64)
    xavier_stack = np.array(xavier_grads, dtype=np.float64)
    zeros_symmetry_std = np.mean(np.std(zeros_stack, axis=0))
    xavier_symmetry_std = np.mean(np.std(xavier_stack, axis=0))

    grad_plot_path = out_dir / "q2_9_gradients_zeros_vs_xavier.png"
    loss_plot_path = out_dir / "q2_9_loss_zeros_vs_xavier.png"
    plot_gradients(zeros_grads, xavier_grads, grad_plot_path)
    plot_loss(zeros_loss, xavier_loss, loss_plot_path)

    wandb.log(
        {
            "q2_9/gradient_plot": wandb.Image(str(grad_plot_path)),
            "q2_9/loss_plot": wandb.Image(str(loss_plot_path)),
        }
    )

    wandb.summary["q2_9/zeros_mean_neuron_grad_std"] = float(zeros_symmetry_std)
    wandb.summary["q2_9/xavier_mean_neuron_grad_std"] = float(xavier_symmetry_std)
    wandb.summary["q2_9/explanation"] = (
        "With zero initialization, neurons in a layer stay symmetric and receive identical updates, "
        "so they learn redundant features. Xavier breaks symmetry by starting neurons differently."
    )

    print(f"Saved: {grad_plot_path}")
    print(f"Saved: {loss_plot_path}")
    print("Q2.9 done. Check W&B for gradient line plots and uploaded images.")

    run.finish()


if __name__ == "__main__":
    main()
