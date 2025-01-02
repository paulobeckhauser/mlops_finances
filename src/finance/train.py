from pathlib import Path
from data import get_training_data
from models import get_model
from tqdm import tqdm

def train_model(model_name, preprocessed_file: Path, **kwargs):
    # Load training and testing data
    X_train, X_test, y_train, y_test = get_training_data(preprocessed_file)

    # Dynamically set the input size for deep learning models
    if model_name == "deep_learning":
        kwargs["input_size"] = X_train.shape[1]  # Set input_size to the number of features

    # Get the model
    model = get_model(model_name, **kwargs)

    if model_name in ["random_forest", "logistic_regression"]:
        # Train scikit-learn models
        model.fit(X_train, y_train)
        y_pred_labels = model.predict(X_test)  # Directly predict class labels

    elif model_name == "deep_learning":
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from torch.nn.functional import cross_entropy

        # Convert to tensors
        train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                       torch.tensor(y_train.values, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training loop
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Training loop with progress bars
        epochs = 10
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            epoch_loss = 0.0
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = cross_entropy(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Test deep learning model
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_test.values, dtype=torch.float32))  # Raw logits
            y_pred_labels = y_pred.argmax(axis=1).numpy()  # Convert logits to class labels

    # Evaluate the model
    if y_pred_labels is not None:  # Ensure predictions exist
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred_labels)
        print(f"{model_name} Model Accuracy: {accuracy:.4f}")
    else:
        raise RuntimeError("Model did not generate predictions.")


