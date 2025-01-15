from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from finance.data import get_training_data
from finance.model import DeepLearningModel


def train_model(model_name, preprocessed_file: Path, **kwargs):
    """
    Train a machine learning model based on the specified model_name.
    Currently supports deep learning with PyTorch Lightning.
    """
    if model_name == "deep_learning":
        # Load training and testing data
        X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)

        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_test.values, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.long),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Initialize the PyTorch Lightning model
        model = DeepLearningModel(
            input_size=X_train.shape[1],
            num_classes=kwargs.get("num_classes", 2),
            lr=kwargs.get("lr", 0.001),
        )

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
            max_epochs=kwargs.get("epochs", 10),
            callbacks=[checkpoint_callback],
            log_every_n_steps=1,
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)
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
