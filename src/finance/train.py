# mypy: disallow-untyped-defs
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from finance.data import get_training_data
from finance.model import DeepLearningModel, get_loaders

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def train_model(
    model_name: str, preprocessed_file: Path, **kwargs: int | float
) -> None:
    """
    Train a machine learning model based on the specified model_name.
    Currently supports deep learning with PyTorch Lightning.
    """
    if model_name == "deep_learning":
        # Load training and testing data
        X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)

        train_loader, val_loader = get_loaders(X_train, X_test, y_train, y_test)

        # Initialize the PyTorch Lightning model
        model = DeepLearningModel(
            input_size=X_train.shape[1],
            num_classes=int(kwargs.get("num_classes", 2)),
            lr=float(kwargs.get("lr", 0.001)),
        )

        # Define the new folder path for saving models
        model_dir = Path("model")
        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Define callbacks (e.g., checkpointing)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        )

        # Initialize PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=int(kwargs.get("epochs", 10)),
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Save the final model weights manually
        save_path = model_dir / "model.pth"
        torch.save(model.state_dict(), save_path)
        log.info(f"Final model save at {save_path}")

    else:
        raise ValueError(
            f"Model {model_name} is not supported with PyTorch Lightning yet."
        )


if __name__ == "__main__":
    # Define model name and file path
    model_name = "deep_learning"  # Options: "random_forest", "logistic_regression", "deep_learning"
    preprocessed_file = Path("data/processed/processed_data.csv")

    # Additional parameters for the model
    model_params = {
        "epochs": 15,  # Number of epochs for deep learning
        "lr": 0.001,  # Learning rate for deep learning
        "num_classes": 3,  # Number of output classes
    }

    # Train the model
    train_model(model_name, preprocessed_file, **model_params)
