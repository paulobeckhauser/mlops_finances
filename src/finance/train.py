# mypy: disallow-untyped-defs
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from finance.data import get_training_data
from finance.model import DeepLearningModel, get_loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train_model(
    model_name: str, preprocessed_file: Path, **kwargs: int | float
) -> None:
    """
    Train a machine learning model based on the specified model_name.
    Currently supports deep learning with PyTorch Lightning.

    Args:
        model_name (str): The name of the model to train (currently supports "deep_learning").
        preprocessed_file (Path): Path to the preprocessed dataset.
        kwargs: Additional parameters for the model, including:
            - epochs: Number of training epochs (default: 10).
            - lr: Learning rate for the optimizer (default: 0.001).
            - num_classes: Number of output classes (default: 2).
    """
    if model_name == "deep_learning":
        # Load training and testing data
        X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)

        # Create DataLoaders for training and validation
        train_loader, val_loader = get_loaders(X_train, X_test, y_train, y_test)

        # Initialize the Deep Learning model
        model = DeepLearningModel(
            input_size=X_train.shape[1],
            num_classes=int(kwargs.get("num_classes", 2)),
            lr=float(kwargs.get("lr", 0.001)),
        )

        # Ensure the model directory exists
        model_dir = Path("model")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Set up checkpoint callback to save the best model
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        )

        # Initialize the PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=int(kwargs.get("epochs", 10)),
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # ** Save the final model weights **
        save_path = model_dir / "model.pth"
        torch.save(model.state_dict(), save_path)  # Save the model's state_dict
        log.info(f"Final model saved at {save_path}")

    else:
        raise ValueError(
            f"Model {model_name} is not supported with PyTorch Lightning yet."
        )


if __name__ == "__main__":
    # Specify the model name and dataset path
    model_name = "deep_learning"  # Options: "random_forest", "logistic_regression", "deep_learning"
    preprocessed_file = Path("data/processed/processed_data.csv")

    # Model parameters for training
    model_params = {
        "epochs": 15,  # Number of epochs for training
        "lr": 0.001,  # Learning rate for the optimizer
        "num_classes": 3,  # Number of output classes
    }

    # Train the model
    train_model(model_name, preprocessed_file, **model_params)
