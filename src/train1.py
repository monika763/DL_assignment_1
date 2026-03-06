import argparse
import numpy as np
import wandb
from tensorflow.keras.datasets import mnist

from ann.neural_network import NeuralNetwork


# -----------------------------
# Load MNIST Dataset
# -----------------------------
def load_dataset():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    return x_train, y_train, x_test, y_test


# -----------------------------
# Log Sample Images
# -----------------------------
def log_sample_images():

    (x_train, y_train), _ = mnist.load_data()

    images = []
    for i in range(10):
        images.append(wandb.Image(x_train[i], caption=f"Label: {y_train[i]}"))

    wandb.log({"sample_images": images})


# -----------------------------
# Train Function
# -----------------------------
def train():

    config = wandb.config

    x_train, y_train, x_test, y_test = load_dataset()

    input_dim = x_train.shape[1]
    output_dim = 10

    hidden_layers = [config.num_neurons] * config.num_hidden_layers

    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        activation_function=config.activation,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        initialization_method=config.initialization,
        loss_function=config.loss_function
    )

    for epoch in range(config.epochs):

        train_loss, train_acc = model.train_epoch(
            x_train,
            y_train,
            batch_size=config.batch_size
        )

        val_loss, val_acc = model.evaluate(x_test, y_test)

        # gradient norm (Q2.4)
        grad_norm = np.linalg.norm(model.grads["W1"])

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "grad_norm_layer1": grad_norm,
            "optimizer": config.optimizer,
            "loss_function": config.loss_function
        })

    # -----------------------------
    # Confusion Matrix (Q2.8)
    # -----------------------------

    preds = model.predict(x_test)

    wandb.log({
        "confusion_matrix":
            wandb.plot.confusion_matrix(
                preds=preds,
                y_true=y_test
            )
    })


# -----------------------------
# Main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--num_neurons", type=int, default=128)

    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--initialization", type=str, default="xavier")

    parser.add_argument("--loss_function", type=str, default="cross_entropy")

    args = parser.parse_args()

    wandb.init(project="da6401_assignment1")

    # -----------------------------
    # Log Sample Images (Q2.1)
    # -----------------------------
    log_sample_images()

    # -----------------------------
    # Train Model
    # -----------------------------
    train()


if __name__ == "__main__":
    main()