from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

from ann.neural_network import NeuralNetwork
from train import load_dataset, parse_neuron_config, train_val_split


def get_best_run(entity: str, project: str, metric_key: str) -> wandb.apis.public.Run:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    valid = []
    for r in runs:
        v = r.summary.get(metric_key, None)
        if v is not None:
            try:
                valid.append((float(v), r))
            except Exception:
                pass

    if not valid:
        raise ValueError(f"No runs found with metric '{metric_key}' in {entity}/{project}")

    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_creative_errors(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    cm: np.ndarray,
    out_bar: Path,
    out_grid: Path,
) -> None:
    # Top confused class pairs (off-diagonal)
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))

    if not pairs:
        print("No misclassifications found.")
        return

    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:10]

    # Bar chart: top confusion pairs
    labels = [f"{t}->{p}" for t, p, _ in top_pairs]
    counts = [c for _, _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts)
    ax.set_title("Top Confused Class Pairs (Creative Error View)")
    ax.set_xlabel("True -> Predicted")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_bar, dpi=220)
    plt.close(fig)

    # Grid of failures for worst pair
    t_cls, p_cls, _ = top_pairs[0]
    idx = np.where((y_test == t_cls) & (y_pred == p_cls))[0]
    idx = idx[:25]

    if idx.size == 0:
        return

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.ravel()
    for k in range(25):
        axes[k].axis("off")
        if k < len(idx):
            image = x_test[idx[k]].reshape(28, 28)
            conf = y_prob[idx[k], p_cls]
            axes[k].imshow(image, cmap="gray")
            axes[k].set_title(f"T:{t_cls} P:{p_cls}\nconf:{conf:.2f}", fontsize=8)

    fig.suptitle(f"Failure Gallery for Worst Pair: {t_cls} -> {p_cls}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_grid, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--metric_key", type=str, default="test/accuracy")
    parser.add_argument("--out_dir", type=str, default="src/models")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--log_project", type=str, default="mnist-ann-analysis")
    parser.add_argument("--log_run_name", type=str, default="q2_8_error_analysis")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_run = get_best_run(args.entity, args.project, args.metric_key)
    cfg = best_run.config

    dataset = str(cfg.get("dataset", "mnist"))
    num_hidden_layers = int(cfg.get("num_hidden_layers", 3))
    num_neurons = parse_neuron_config(cfg.get("num_neurons", "128"), num_hidden_layers)

    initialization_method = str(cfg.get("initialization_method", "xavier"))
    activation_function = str(cfg.get("activation_function", "relu"))
    optimizer = str(cfg.get("optimizer", "adam"))
    learning_rate = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("epochs", 10))
    batch_size = int(cfg.get("batch_size", 64))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    val_ratio = float(cfg.get("val_ratio", 0.1))
    random_seed = int(cfg.get("random_seed", 42))

    np.random.seed(random_seed)

    (x_train_full, y_train_full), (x_test, y_test) = load_dataset(dataset)
    x_train_full = x_train_full.reshape(x_train_full.shape[0], -1).astype(np.float64) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float64) / 255.0
    y_train_full = y_train_full.astype(int)
    y_test = y_test.astype(int)

    x_train, y_train, x_val, y_val = train_val_split(
        x_train_full, y_train_full, val_ratio=val_ratio, random_seed=random_seed
    )

    model = NeuralNetwork(
        input_dim=x_train.shape[1],
        output_dim=10,
        num_neurons=num_neurons,
        num_hidden_layers=num_hidden_layers,
        initialization_method=initialization_method,
        activation_function=activation_function,
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    history = model.fit(x_train, y_train, x_val=x_val, y_val=y_val, verbose=True)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    train_acc_final = history.train_accuracy[-1]

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))

    cm_path = out_dir / "q2_8_confusion_matrix.png"
    err_bar_path = out_dir / "q2_8_top_confusions.png"
    err_grid_path = out_dir / "q2_8_failure_gallery.png"

    plot_confusion_matrix(cm, cm_path)
    plot_creative_errors(x_test, y_test, y_pred, y_prob, cm, err_bar_path, err_grid_path)

    if args.log_to_wandb:
        run = wandb.init(
            project=args.log_project,
            entity=args.entity,
            mode=args.wandb_mode,
            name=args.log_run_name,
            config={
                "source_project": args.project,
                "metric_key": args.metric_key,
                "best_run_id": best_run.id,
                "best_run_name": best_run.name,
            },
        )

        wandb.log(
            {
                "q2_8/confusion_matrix_image": wandb.Image(str(cm_path)),
                "q2_8/top_confusions_image": wandb.Image(str(err_bar_path)),
                "q2_8/failure_gallery_image": wandb.Image(str(err_grid_path)),
                "q2_8/confusion_matrix_plot": wandb.plot.confusion_matrix(
                    y_true=y_test.tolist(),
                    preds=y_pred.tolist(),
                ),
            }
        )

        # Log top-10 confusion pairs as a table.
        pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] > 0:
                    pairs.append((i, j, int(cm[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:10]

        pair_table = wandb.Table(columns=["true_class", "pred_class", "count"])
        for t, p, c in top_pairs:
            pair_table.add_data(int(t), int(p), int(c))
        wandb.log({"q2_8/top_confusion_pairs": pair_table})

        wandb.summary["best_run_id"] = best_run.id
        wandb.summary["best_run_name"] = best_run.name
        wandb.summary["train_accuracy_final"] = float(train_acc_final)
        wandb.summary["test_accuracy"] = float(test_acc)
        wandb.summary["test_loss"] = float(test_loss)
        wandb.summary["num_test_samples"] = int(len(y_test))

        if top_pairs:
            t0, p0, c0 = top_pairs[0]
            wandb.summary["worst_pair_true"] = int(t0)
            wandb.summary["worst_pair_pred"] = int(p0)
            wandb.summary["worst_pair_count"] = int(c0)

        run.finish()

    print(f"Best W&B run: {best_run.name} ({best_run.id})")
    print(f"Final train accuracy: {train_acc_final:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {err_bar_path}")
    print(f"Saved: {err_grid_path}")
    print("\nInterpretation:")
    print("Confusion matrix shows class-wise errors. The failure gallery highlights systematic mistakes on the most confused class pair.")


if __name__ == "__main__":
    main()
