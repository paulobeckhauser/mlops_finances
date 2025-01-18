# mypy: disallow-untyped-defs
import numpy as np
import pandas as pd
import pytest
from pytorch_lightning import Trainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from finance.model import DeepLearningModel, get_loaders, get_model


def test_get_model() -> None:
    random_forest = get_model("random_forest")
    assert type(random_forest) is RandomForestClassifier

    logistic_regression = get_model("logistic_regression")
    assert type(logistic_regression) is LogisticRegression

    with pytest.raises(ValueError) as error:
        get_model("deep_learning")
    assert (
        str(error.value) == "For deep learning models, 'input_size' must be specified."
    )

    with pytest.raises(ValueError) as error:
        get_model("unknown")
    assert str(error.value) == "Unknown model name: unknown"

    deep_learning = get_model("deep_learning", input_size=1)
    assert type(deep_learning) is DeepLearningModel

    #

    x = np.linspace(0.0, np.pi * 0.5, 160)
    y = np.sin(x)

    X_train = pd.DataFrame(data={"values": x[:128]})
    X_test = pd.DataFrame(data={"values": x[128:]})
    y_train = pd.Series((y[:128] > 0.5).squeeze(), name="values")
    y_test = pd.Series((y[128:] > 0.5).squeeze(), name="values")

    train_loader, val_loader = get_loaders(X_train, X_test, y_train, y_test)
    trainer = Trainer(max_epochs=10)

    trainer.fit(deep_learning, train_loader, val_loader)
