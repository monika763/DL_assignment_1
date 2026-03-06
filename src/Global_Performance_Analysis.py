from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_from_run(run, keys: list[str]):
    # 1) Try summary first (fast + final metric in most runs)
    for key in keys:
        v = _to_float(run.summary.get(key))
        if v is not None:
            return v

    # 2) Fallback: read history and take LAST value (not max)
    for key in keys:
        try:
            history = run.history(keys=[key], pandas=False)
        except Exception:
            continue

        values = []
        for row in history:
            v = _to_float(row.get(key))
            if v is not None:
                values.append(v)

        if values:
            return values[-1]  # IMPORTANT: final value, not max

    return None


def _get_runs(entity: str, project: str):
    api = wandb.Api()
    return api.runs(f"{entity}/{project}")


def main():
    parser = argparse.ArgumentParser(
        description="Q2.7: Overlay Training vs Test Accuracy across hyperparameter runs"
    )
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)

    parser.add_argument("--train_high", type=float, default=0.95)
    parser.add_argument("--gap_threshold", type=float, default=0.03)
    parser.add_argument("--only_sweep", action="store_true", help="Include only sweep runs")
    parser.add_argument("--out", type=str, default="src/models/train_vs_test_overlay.png")
    args = parser.parse_args()

    runs = list(_get_runs(args.entity, args.project))
    runs = sorted(runs, key=lambda r: str(getattr(r, "created_at", "")))

    names, train_accs, test_accs = [], [], []

    for run in runs:
        if run.state != "finished":
            continue
        if args.only_sweep and run.sweep is None:
            continue

        train_acc = _metric_from_run(run, ["train/accuracy", "train_accuracy"])
        test_acc = _metric_from_run(run, ["test/accuracy", "test_accuracy"])

        if train_acc is None or test_acc is None:
            continue

        names.append(f"{run.name or run.id} ({run.id[:6]})")
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    if not names:
        print("No valid runs found with both train and test accuracy.")
        return

    train_arr = np.array(train_accs, dtype=np.float64)
    test_arr = np.array(test_accs, dtype=np.float64)
    gap = train_arr - test_arr

    overfit_idx = np.where((train_arr >= args.train_high) & (gap >= args.gap_threshold))[0]

    x = np.arange(len(names))
    fig = plt.figure(figsize=(14, 6))
    plt.plot(x, train_arr, marker="o", label="Training Accuracy")
    plt.plot(x, test_arr, marker="s", label="Test Accuracy")

    if overfit_idx.size > 0:
        plt.scatter(overfit_idx, train_arr[overfit_idx], color="red", s=70, label="High-train / Low-test")
        plt.scatter(overfit_idx, test_arr[overfit_idx], color="red", s=70)

    plt.xticks(x, names, rotation=90)
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Training vs Test Accuracy Across Hyperparameter Runs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved overlay plot: {out_path}")
    print(f"Total runs used: {len(names)}")
    print(f"Overfitting candidates (train >= {args.train_high} and gap >= {args.gap_threshold}):")

    if overfit_idx.size == 0:
        print("None with current threshold.")
    else:
        for i in overfit_idx:
            print(f"- {names[i]} | train={train_arr[i]:.4f} | test={test_arr[i]:.4f} | gap={gap[i]:.4f}")

    print("\nInterpretation:")
    print("A high training accuracy but much lower test accuracy indicates overfitting (poor generalization).")


if __name__ == "__main__":
    main()
