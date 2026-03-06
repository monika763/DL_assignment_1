from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64).reshape(-1)
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def _load_openml_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset_map = {
        "fashion_mnist": "Fashion-MNIST",
        "mnist": "mnist_784",
    }
    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Supported: ['fashion_mnist', 'mnist']"
        )

    x, y = fetch_openml(
        dataset_map[dataset_name],
        version=1,
        return_X_y=True,
        as_frame=False,
    )
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    return x, y


def load_dataset(
    dataset_name: str = "mnist",
    validation_split: float = 0.1,
    random_seed: int = 42,
    normalize: bool = True,
    flatten: bool = True,
    one_hot: bool = True,
) -> DatasetSplits:
    """
    Load dataset and prepare train/val/test splits.

    Behavior:
    - Uses OpenML dataset source.
    - Keeps canonical split as first 60k samples (train+val) and last 10k (test).
    - Splits train+val into train and validation using stratified split.
    """
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be in the range (0, 1).")

    x, y = _load_openml_dataset(dataset_name)
    x_train_val, y_train_val = x[:60000], y[:60000]
    x_test, y_test = x[60000:], y[60000:]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_split,
        random_state=random_seed,
        stratify=y_train_val,
    )

    if not flatten:
        x_train = x_train.reshape(-1, 28, 28)
        x_val = x_val.reshape(-1, 28, 28)
        x_test = x_test.reshape(-1, 28, 28)

    if normalize:
        x_train = x_train / 255.0
        x_val = x_val / 255.0
        x_test = x_test / 255.0

    if one_hot:
        y_train = one_hot_encode(y_train, num_classes=10)
        y_val = one_hot_encode(y_val, num_classes=10)
        y_test = one_hot_encode(y_test, num_classes=10)

    return DatasetSplits(
        x_train=x_train.astype(np.float32),
        y_train=y_train,
        x_val=x_val.astype(np.float32),
        y_val=y_val,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
    )


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    num_samples = x.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]
        yield x[batch_indices], y[batch_indices]
