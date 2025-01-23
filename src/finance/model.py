# mypy: disallow-untyped-defs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
import pandas as pd
from typing import Any, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def get_model(model_name: str, **kwargs: int) -> Any:
    """
    Return the model object based on the model_name.
    Arguments:
        model_name (str): Name of the model to be retrieved.
        kwargs: Additional parameters to configure the model.
    Returns:
        model: The specified model instance.
    """
    if model_name == "random_forest":
        # Random Forest model
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),  # Default to 100 estimators
            max_depth=kwargs.get("max_depth", None),
            random_state=kwargs.get("random_state", 42),
        )
    elif model_name == "logistic_regression":
        # Logistic Regression model
        return LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),  # Default to 1000 iterations
            random_state=kwargs.get("random_state", 42),
        )
    elif model_name == "deep_learning":
        input_size = kwargs.get("input_size")
        if input_size is None:
            raise ValueError("For deep learning models, 'input_size' must be specified.")
        return DeepLearningModel(
            input_size=input_size, num_classes=kwargs.get("num_classes", 2)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_loaders(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation datasets.
    """
    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

class DeepLearningModel(pl.LightningModule):
    """
    Custom Deep Learning model using PyTorch Lightning.
    """

    def __init__(self, input_size: int, num_classes: int, lr: float = 0.001) -> None:
        super(DeepLearningModel, self).__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits
        return x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        """
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """
        Validation step.
        """
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """
        Test step.
        """
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


