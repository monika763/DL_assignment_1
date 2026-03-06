from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from train import load_dataset, train_val_split


# Three base configurations chosen from strong MNIST settings in this repo.
BASE_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "cfg1_3x512_relu_adam_he",
        "num_hidden_layers": 3,
        "num_neurons": 512,
        "activation_function": "relu",
        "optimizer": "adam",
        "initialization_method": "he",
        "learning_rate": 0.003,
        "batch_size": 64,
        "epochs": 10,
        "weight_decay": 0.0001,
        "random_seed": 42,
    },
    {
        "name": "cfg2_4x512_relu_adam_xavier",
        "num_hidden_layers": 4,
        "num_neurons": 512,
        "activation_function": "relu",
        "optimizer": "adam",
        "initialization_method": "xavier",
        "learning_rate": 0.0005,
        "batch_size": 64,
        "epochs": 10,
        "weight_decay": 0.0005,
        "random_seed": 123,
    },
    {
        "name": "cfg3_4x512_tanh_nesterov_xavier",
        "num_hidden_layers": 4,
        "num_neurons": 512,
        "activation_function": "tanh",
        "optimizer": "nesterov",
        "initialization_method": "xavier",
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 15,
        "weight_decay": 0.0,
        "random_seed": 7,
    },
]


def _metric_from_summary(run: wandb.apis.public.Run, key: str) -> float | None:
    value = run.summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def best_mnist_signature(entity: str, project: str) -> tuple[dict[str, Any] | None, float | None]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    candidates: list[tuple[float, wandb.apis.public.Run]] = []
    for run in runs:
        dataset = str(run.config.get("dataset", "")).lower()
        if dataset != "mnist":
            continue
        score = _metric_from_summary(run, "test/accuracy")
        if score is None:
            score = _metric_from_summary(run, "test_accuracy")
        if score is None:
            continue
        candidates.append((score, run))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_run = candidates[0]

    signature = {
        "num_hidden_layers": int(best_run.config.get("num_hidden_layers", -1)),
        "num_neurons": int(best_run.config.get("num_neurons", -1)),
        "activation_function": str(best_run.config.get("activation_function", "")),
        "optimizer": str(best_run.config.get("optimizer", "")),
    }
    return signature, float(best_score)


def run_single_config(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    np.random.seed(int(config["random_seed"]))

    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=int(config["num_neurons"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        initialization_method=str(config["initialization_method"]),
        activation_function=str(config["activation_function"]),
        optimizer=str(config["optimizer"]),
        learning_rate=float(config["learning_rate"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        weight_decay=float(config["weight_decay"]),
        random_seed=int(config["random_seed"]),
    )

    history = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, verbose=True)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    train_final = float(history.train_accuracy[-1])
    val_final = float(history.val_accuracy[-1]) if history.val_accuracy else float("nan")

    return {
        **config,
        "final_train_accuracy": train_final,
        "final_val_accuracy": val_final,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "gen_gap_train_minus_test": float(train_final - test_acc),
    }


def save_bar_plot(results: list[dict[str, Any]], out_path: Path) -> None:
    names = [r["name"] for r in results]
    test_scores = [r["test_accuracy"] for r in results]

    fig = plt.figure(figsize=(10, 5))
    bars = plt.bar(names, test_scores)
    plt.ylabel("Test Accuracy")
    plt.xlabel("Configuration")
    plt.title("Fashion-MNIST Transfer Challenge: 3 Configurations")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    for b, score in zip(bars, test_scores):
        plt.text(b.get_x() + b.get_width() / 2, score + 0.005, f"{score:.4f}", ha="center", fontsize=9)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def build_run_plan(total_runs: int, seed_step: int) -> list[dict[str, Any]]:
    if total_runs <= 0:
        raise ValueError("--total_runs must be >= 1")

    plan: list[dict[str, Any]] = []
    base_count = len(BASE_CONFIGS)
    for idx in range(total_runs):
        base = dict(BASE_CONFIGS[idx % base_count])
        repeat_round = idx // base_count

        base_name = str(base["name"])
        base_seed = int(base["random_seed"])
        run_seed = base_seed + repeat_round * seed_step

        base["base_name"] = base_name
        base["random_seed"] = run_seed
        base["name"] = f"{base_name}_run{idx + 1}"
        plan.append(base)

    return plan


def summarize_by_base(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row["base_name"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for base_name, rows in grouped.items():
        test_scores = np.array([float(r["test_accuracy"]) for r in rows], dtype=np.float64)
        val_scores = np.array([float(r["final_val_accuracy"]) for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "base_name": base_name,
                "runs": len(rows),
                "mean_test_accuracy": float(np.mean(test_scores)),
                "std_test_accuracy": float(np.std(test_scores)),
                "mean_val_accuracy": float(np.mean(val_scores)),
                "best_test_accuracy": float(np.max(test_scores)),
            }
        )

    summary_rows.sort(key=lambda x: x["mean_test_accuracy"], reverse=True)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2.10 Fashion-MNIST Transfer Challenge (3-config budget)")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--source_project", type=str, default="mnist-ann")
    parser.add_argument("--log_project", type=str, default="fashion-transfer-challenge")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--run_name", type=str, default="q2_10_transfer_3_configs")
    parser.add_argument("--total_runs", type=int, default=3, help="Total training runs to execute (set 25 for your request).")
    parser.add_argument("--seed_step", type=int, default=17, help="Seed increment per repeat cycle.")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="src/models")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.log_project,
        entity=args.entity,
        mode=args.wandb_mode,
        name=args.run_name,
        config={
            "dataset": "fashion_mnist",
            "base_num_configs": 3,
            "total_runs": args.total_runs,
            "source_project": args.source_project,
            "val_ratio": args.val_ratio,
            "split_seed": args.split_seed,
        },
    )

    (x_train_full, y_train_full), (x_test, y_test) = load_dataset("fashion_mnist")
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    y_test = y_test.astype(int)

    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full,
        y_train_full,
        val_ratio=float(args.val_ratio),
        random_seed=int(args.split_seed),
    )

    run_plan = build_run_plan(total_runs=int(args.total_runs), seed_step=int(args.seed_step))

    results: list[dict[str, Any]] = []
    for idx, config in enumerate(run_plan, start=1):
        print(f"\n[{idx}/{len(run_plan)}] Running {config['name']}")
        result = run_single_config(config, x_train, y_train, x_val, y_val, x_test, y_test)
        results.append(result)
        wandb.log(
            {
                "config_index": idx,
                "config_name": result["name"],
                "final_train_accuracy": result["final_train_accuracy"],
                "final_val_accuracy": result["final_val_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "test_loss": result["test_loss"],
                "gen_gap_train_minus_test": result["gen_gap_train_minus_test"],
            },
            step=idx,
        )

    results.sort(key=lambda r: r["test_accuracy"], reverse=True)
    best_fashion = results[0]
    base_summary = summarize_by_base(results)

    bar_path = out_dir / "q2_10_fashion_transfer_accuracies.png"
    save_bar_plot(results, bar_path)
    wandb.log({"q2_10/fashion_accuracy_bar": wandb.Image(str(bar_path))})

    table = wandb.Table(
        columns=[
            "rank",
            "name",
            "num_hidden_layers",
            "num_neurons",
            "activation_function",
            "optimizer",
            "initialization_method",
            "learning_rate",
            "batch_size",
            "epochs",
            "weight_decay",
            "final_train_accuracy",
            "final_val_accuracy",
            "test_accuracy",
            "test_loss",
            "gen_gap_train_minus_test",
        ]
    )
    for rank, r in enumerate(results, start=1):
        table.add_data(
            rank,
            r["name"],
            r["num_hidden_layers"],
            r["num_neurons"],
            r["activation_function"],
            r["optimizer"],
            r["initialization_method"],
            r["learning_rate"],
            r["batch_size"],
            r["epochs"],
            r["weight_decay"],
            r["final_train_accuracy"],
            r["final_val_accuracy"],
            r["test_accuracy"],
            r["test_loss"],
            r["gen_gap_train_minus_test"],
        )
    wandb.log({"q2_10/results_table": table})

    summary_table = wandb.Table(
        columns=[
            "base_name",
            "runs",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_val_accuracy",
            "best_test_accuracy",
        ]
    )
    for row in base_summary:
        summary_table.add_data(
            row["base_name"],
            row["runs"],
            row["mean_test_accuracy"],
            row["std_test_accuracy"],
            row["mean_val_accuracy"],
            row["best_test_accuracy"],
        )
    wandb.log({"q2_10/base_config_summary": summary_table})

    mnist_best_sig, mnist_best_acc = best_mnist_signature(args.entity, args.source_project)
    if mnist_best_sig is not None:
        same_as_mnist_best = (
            int(best_fashion["num_hidden_layers"]) == int(mnist_best_sig["num_hidden_layers"])
            and int(best_fashion["num_neurons"]) == int(mnist_best_sig["num_neurons"])
            and str(best_fashion["activation_function"]).lower() == str(mnist_best_sig["activation_function"]).lower()
            and str(best_fashion["optimizer"]).lower() == str(mnist_best_sig["optimizer"]).lower()
        )
        wandb.summary["mnist_best_test_accuracy"] = float(mnist_best_acc)
        wandb.summary["mnist_best_signature"] = str(mnist_best_sig)
        wandb.summary["fashion_best_matches_mnist_best"] = bool(same_as_mnist_best)
    else:
        same_as_mnist_best = None

    wandb.summary["fashion_best_config"] = best_fashion["name"]
    wandb.summary["fashion_best_test_accuracy"] = float(best_fashion["test_accuracy"])
    wandb.summary["fashion_best_generalization_gap"] = float(best_fashion["gen_gap_train_minus_test"])
    wandb.summary["total_runs_executed"] = int(len(results))
    if base_summary:
        wandb.summary["best_base_config_by_mean_test"] = base_summary[0]["base_name"]
        wandb.summary["best_base_mean_test_accuracy"] = float(base_summary[0]["mean_test_accuracy"])

    print("\n=== Q2.10 Results ===")
    print(f"Total runs executed: {len(results)}")
    for r in results:
        print(
            f"{r['name']}: test_acc={r['test_accuracy']:.4f}, "
            f"val_acc={r['final_val_accuracy']:.4f}, train_acc={r['final_train_accuracy']:.4f}"
        )

    print("\nBase-config summary (mean over repeated runs):")
    for row in base_summary:
        print(
            f"{row['base_name']}: mean_test={row['mean_test_accuracy']:.4f}, "
            f"std={row['std_test_accuracy']:.4f}, best={row['best_test_accuracy']:.4f}, runs={row['runs']}"
        )
    print(f"Best Fashion-MNIST config: {best_fashion['name']} ({best_fashion['test_accuracy']:.4f})")

    if same_as_mnist_best is True:
        print("Did MNIST-best also win on Fashion-MNIST? YES")
    elif same_as_mnist_best is False:
        print("Did MNIST-best also win on Fashion-MNIST? NO")
    else:
        print("Did MNIST-best also win on Fashion-MNIST? Could not determine (no MNIST run found via API).")

    print(
        "Why dataset complexity matters: Fashion-MNIST has higher intra-class variation and inter-class similarity, "
        "so optimization/regularization choices that work for digits may not transfer perfectly."
    )

    run.finish()
if __name__ == "__main__":
    main()
