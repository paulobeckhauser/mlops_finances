from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch
import pytorch_lightning as pl

def get_model(model_name, **kwargs):
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
            random_state=kwargs.get("random_state", 42)
        )
    elif model_name == "logistic_regression":
        # Logistic Regression model
        return LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),  # Default to 1000 iterations
            random_state=kwargs.get("random_state", 42)
        )
    elif model_name == "deep_learning":
        # Custom Deep Learning model
        input_size = kwargs.get("input_size")
        if input_size is None:
            raise ValueError("For deep learning models, 'input_size' must be specified.")
        return DeepLearningModel(input_size=input_size, num_classes=kwargs.get("num_classes", 2))
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# class DeepLearningModel(nn.Module):
#     """
#     Custom deep learning model for classification.
#     """
#     def __init__(self, input_size, num_classes):
#         super(DeepLearningModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)  # Input to hidden layer
#         self.fc2 = nn.Linear(128, 64)         # Hidden to hidden layer
#         self.fc3 = nn.Linear(64, num_classes) # Hidden to output layer
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # Logits, no activation
#         return x

class DeepLearningModel(pl.LightningModule):
    """
    PyTorch Lightning model for classification.
    """
    def __init__(self, input_size, num_classes, lr=0.001):
        super(DeepLearningModel, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.lr = lr

        # Define a softmax for evaluation purposes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits (raw predictions)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = nn.CrossEntropyLoss()(y_pred, y_batch)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)