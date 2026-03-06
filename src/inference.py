from __future__ import annotations

import argparse

from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD
from utils.data_loader import load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference/evaluation on test split.")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "mnist"])
    parser.add_argument("--model_path", type=str, default="models/best_model.npz")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data = load_dataset(
        dataset_name=args.dataset,
        validation_split=0.1,
        normalize=True,
        flatten=True,
        one_hot=True,
    )

    # Optimizer is required by the model constructor but is unused in inference.
    model = NeuralNetwork.load(args.model_path, optimizer=SGD(lr=0.0))
    test_loss, test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
