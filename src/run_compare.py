"""Run a W&B sweep where only optimizer changes (single agent, 5 configs).

Only train/loss and val/loss are logged to W&B.
"""

from __future__ import annotations

import argparse
from typing import Any

import wandb

from train import compare_optimizer_convergence


def parse_optimizers(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one optimizer must be provided.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W&B sweep: only optimizer changes, all other config fixed.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--target_val_accuracy", type=float, default=0.90)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--optimizers",
        type=str,
        default="sgd,momentum,nesterov,rmsprop,adam",
        help="Comma-separated list. Use 5 values for 5 configurations.",
    )
    parser.add_argument("--agent_count", type=int, default=5, help="Number of runs executed by a single W&B agent.")

    parser.add_argument("--wandb_project", type=str, default="mnist-ann")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--sweep_name", type=str, default="optimizer-only-sweep-3x128-relu")
    return parser


def _train_one_config(base_cfg: dict[str, Any]) -> None:
    run = wandb.init(project=base_cfg["wandb_project"], entity=base_cfg["wandb_entity"], mode=base_cfg["wandb_mode"])
    cfg = dict(run.config)
    optimizer_name = str(cfg["optimizer"])

    run.name = f"opt-{optimizer_name}"

    result = compare_optimizer_convergence(
        dataset=str(base_cfg["dataset"]),
        optimizers=(optimizer_name,),
        num_hidden_layers=3,
        num_neurons=128,
        activation_function="relu",
        learning_rate=float(base_cfg["learning_rate"]),
        epochs=int(base_cfg["epochs"]),
        batch_size=int(base_cfg["batch_size"]),
        weight_decay=float(base_cfg["weight_decay"]),
        random_seed=int(base_cfg["random_seed"]),
        target_val_accuracy=float(base_cfg["target_val_accuracy"]),
        verbose=False,
    )

    history = result["histories"][optimizer_name]
    for epoch_idx in range(int(base_cfg["epochs"])):
        metrics: dict[str, float | int] = {
            "epoch": epoch_idx + 1,
            "train/loss": history.train_loss[epoch_idx],
        }
        if history.val_loss:
            metrics["val/loss"] = history.val_loss[epoch_idx]
        wandb.log(metrics, step=epoch_idx + 1)

    wandb.summary["architecture"] = "3_hidden_layers_x_128_neurons_relu"
    wandb.summary["optimizer"] = optimizer_name
    wandb.summary["final_train_loss"] = history.train_loss[-1]
    wandb.summary["final_val_loss"] = history.val_loss[-1] if history.val_loss else None
    wandb.finish()


def main() -> None:
    args = build_parser().parse_args()
    optimizer_values = parse_optimizers(args.optimizers)

    if args.agent_count > len(optimizer_values):
        raise ValueError(
            f"agent_count={args.agent_count} but only {len(optimizer_values)} optimizer values provided."
        )

    sweep_config = {
        "name": args.sweep_name,
        "method": "grid",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {"optimizer": {"values": optimizer_values}},
    }

    base_cfg = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "target_val_accuracy": args.target_val_accuracy,
        "random_seed": args.random_seed,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    print(f"Created sweep: {sweep_id}")
    wandb.agent(sweep_id, function=lambda: _train_one_config(base_cfg), count=args.agent_count)


if __name__ == "__main__":
    main()