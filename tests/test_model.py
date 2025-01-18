# mypy: disallow-untyped-defs
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from finance.model import DeepLearningModel, get_model, get_loaders

def test_get_model() -> None:
    # Test model initialization
    random_forest = get_model("random_forest")
    assert isinstance(random_forest, RandomForestClassifier)

    logistic_regression = get_model("logistic_regression")
    assert isinstance(logistic_regression, LogisticRegression)

    deep_learning = get_model("deep_learning", input_size=1)
    assert isinstance(deep_learning, DeepLearningModel)

    # Test invalid model name
    try:
        get_model("invalid_model_name")
        assert False, "Expected ValueError for invalid model name"
    except ValueError as e:
        assert str(e) == "Unknown model name: invalid_model_name"

    # Create synthetic data
    x = np.linspace(0.0, np.pi * 0.5, 160)
    y = np.sin(x)

    # Convert to tensors
    X_train = torch.tensor(x[:128], dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(x[128:], dtype=torch.float32).unsqueeze(1)

    # Discretize y into class indices
    y_train = (torch.tensor(y[:128], dtype=torch.float32) > 0.5).long()
    y_test = (torch.tensor(y[128:], dtype=torch.float32) > 0.5).long()

    # Get data loaders
    train_loader, val_loader = get_loaders(X_train, X_test, y_train, y_test)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Train the model
    trainer = Trainer(max_epochs=10)
    trainer.fit(deep_learning, train_loader, val_loader)

    # Validate training metrics
    assert "train_loss" in trainer.callback_metrics
    