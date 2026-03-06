# DA6401 Assignment 1

Roll No: MA24M016

This repository contains the code for DA6401 Assignment 1 (feedforward neural
network from scratch, training, evaluation, and experiment tracking).

## Project Structure

```text
da6401_assignment_1/
  README.md
  requirements.txt
  models/
  notebooks/
    wandb_demo.py
  src/
    train.py
    inference.py
    ann/
      activations.py
      neural_layers.py
      neural_network.py
      objective_functions.py
      optimizers.py
    utils/
      data_loader.py
```

## Setup

1. Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Commands

Train model on Fashion-MNIST (default):

```powershell
python src/train.py
```

Train on MNIST:

```powershell
python src/train.py --dataset mnist
```

Run inference:

```powershell
python src/inference.py
```

Run W&B demo/experiment script:

```powershell
python notebooks/wandb_demo.py
```

## Step-by-Step Workflow

1. Install dependencies:
```powershell
pip install -r requirements.txt
```
2. Start training (Fashion-MNIST):
```powershell
python src/train.py --epochs 10 --batch_size 64 --optimizer adam
```
3. Saved best model:
`models/best_model.npz`
4. Evaluate test accuracy using saved model:
```powershell
python src/inference.py --dataset fashion_mnist --model_path models/best_model.npz
```
5. Optional: run W&B tracked experiment:
```powershell
python notebooks/wandb_demo.py
```

## Implemented Files (Current Status)

1. `src/utils/data_loader.py` - dataset loading + batching
2. `src/ann/activations.py` - activation functions + derivatives + softmax
3. `src/ann/objective_functions.py` - cross-entropy + accuracy
4. `src/ann/neural_layers.py` - dense layer forward/backward
5. `src/ann/optimizers.py` - SGD, Momentum, NAG, RMSProp, Adam
6. `src/ann/neural_network.py` - model forward/backward/train/evaluate/save/load
7. `src/train.py` - training pipeline + CLI + optional W&B logging
8. `src/inference.py` - model loading + test evaluation
9. `notebooks/wandb_demo.py` - simple W&B-enabled training run

## Data Loader (Step 3 Completed)

Implemented in `src/utils/data_loader.py`:

- `load_dataset(...)`
  - Supports: `fashion_mnist`, `mnist`
  - Canonical split: first 60k for train+val, last 10k for test
  - Train/val split with stratification
  - Optional normalization, flattening, and one-hot encoding
- `batch_iterator(...)`
  - Mini-batch generator with optional shuffling

Quick usage:

```python
from src.utils.data_loader import load_dataset, batch_iterator

data = load_dataset(dataset_name="mnist", validation_split=0.1)
print(data.x_train.shape, data.y_train.shape)

for x_batch, y_batch in batch_iterator(data.x_train, data.y_train, batch_size=64):
    # training step
    pass
```
