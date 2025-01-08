from pathlib import Path
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from data import get_training_data
from model import DeepLearningModel

def evaluate_model(checkpoint_path: Path, preprocessed_file: Path, **kwargs):
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
    test_loader = DataLoader(test_dataset, batch_size=kwargs.get("batch_size", 32), shuffle=False)

    # Load the model from the checkpoint
    model = DeepLearningModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)  # Forward pass
            preds = torch.argmax(y_pred, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Detailed metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

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
