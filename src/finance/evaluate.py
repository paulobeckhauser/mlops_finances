# mypy: disallow-untyped-defs
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from finance.data import get_training_data
from finance.model import DeepLearningModel


def evaluate_model(
    checkpoint_path: Path, preprocessed_file: Path, **kwargs: int
) -> None:
    """
    Evaluate a trained model on the test dataset.

    Args:
        checkpoint_path (Path): Path to the trained model checkpoint.
        preprocessed_file (Path): Path to the preprocessed dataset.
        **kwargs: Additional arguments like batch size.
    """
    # Load test data
    _, X_test, _, y_test = get_training_data(preprocessed_file)

    # Convert test data to PyTorch tensors
    test_dataset = TensorDataset(
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    test_loader = DataLoader(
        test_dataset, batch_size=kwargs.get("batch_size", 32), shuffle=False
    )

    # Load the model from the checkpoint
    model = DeepLearningModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Determine task type and initialize metrics
    task_type = str(kwargs.get("task", "binary"))
    num_classes = int(kwargs.get("num_classes", 2))
    accuracy_metric = Accuracy(
        task=task_type, num_classes=num_classes, average="macro"  # type: ignore [arg-type]
    )
    precision_metric = Precision(
        task=task_type, num_classes=num_classes, average="macro"  # type: ignore [arg-type]
    )
    recall_metric = Recall(
        task=task_type, num_classes=num_classes, average="macro"  # type: ignore [arg-type]
    )
    f1_metric = F1Score(
        task=task_type, num_classes=num_classes, average="macro"  # type: ignore [arg-type]
    )

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)  # Forward pass
            preds = torch.argmax(y_pred, axis=1)  # type: ignore [call-arg]
            all_preds.append(preds)
            all_labels.append(y_batch)

    # Concatenate all predictions and labels
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Compute metrics
    accuracy = accuracy_metric(preds, preds).item()
    precision = precision_metric(preds, preds).item()
    recall = recall_metric(preds, preds).item()
    f1 = f1_metric(preds, preds).item()

    # Print metrics
    print(f"Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    # Path to the trained model checkpoint
    checkpoint_path = Path("checkpoints/best-checkpoint.ckpt")
    # Path to the preprocessed dataset
    preprocessed_file = Path("data/processed/processed_data.csv")

    # Evaluate the model
    evaluate_model(
        checkpoint_path=checkpoint_path,
        preprocessed_file=preprocessed_file,
        batch_size=32,
    )
