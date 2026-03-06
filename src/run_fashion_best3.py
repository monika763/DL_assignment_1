from __future__ import annotations

import argparse

from train import train_single_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 3 fixed Fashion-MNIST experiments: best1, best2, best3."
    )
    parser.add_argument("--wandb_project", type=str, default="mnist-ann")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    # Common hyperparameters kept SAME across all three runs.
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--initialization_method", type=str, default="he")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base = {
        "dataset": "fashion_mnist",
        "num_hidden_layers": 2,
        "num_neurons": "256",
        "initialization_method": args.initialization_method,
        "activation_function": "relu",
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "val_ratio": args.val_ratio,
        "random_seed": args.random_seed,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_mode": args.wandb_mode,
        "run_sweep": False,
        "sweep_config": "",
        "sweep_count": 0,
    }

    # Vary only architecture + optimizer + activation as requested.
    run_plan = [
        {
            "wandb_run_name": "best1",
            "num_hidden_layers": 4,
            "num_neurons": "256",
            "activation_function": "tanh",
            "optimizer": "rmsprop",
        },
        {
            "wandb_run_name": "best2",
            "num_hidden_layers": 2,
            "num_neurons": "128",
            "activation_function": "relu",
            "optimizer": "adam",
        },
        {
            "wandb_run_name": "best3",
            "num_hidden_layers": 4,
            "num_neurons": "128",
            "activation_function": "relu",
            "optimizer": "nestrov",
        },
    ]

    for idx, override in enumerate(run_plan, start=1):
        cfg = dict(base)
        cfg.update(override)
        run_args = argparse.Namespace(**cfg)
        print(
            f"[{idx}/3] Running {cfg['wandb_run_name']} | "
            f"layers={cfg['num_hidden_layers']} neurons={cfg['num_neurons']} "
            f"act={cfg['activation_function']} opt={cfg['optimizer']} dataset={cfg['dataset']}"
        )
        train_single_run(run_args, sweep_run=False)


if __name__ == "__main__":
    main()
