from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
from src.finance.model import DeepLearningModel  # Import the model class

# Define paths and model parameters
MODEL_PATH = "model/model.pth"
INPUT_SIZE = 4  # Update this based on your dataset's feature size
NUM_CLASSES = 2  # Number of output classes

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
try:
    model = DeepLearningModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))  # Load the model's state_dict
    model.eval()  # Set the model to evaluation mode
except FileNotFoundError:
    raise RuntimeError(f"Model weights not found at {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

@app.get("/")
def read_root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Welcome to the MLOps Finances API!"}

@app.post("/predict/")
def predict(input_data: dict):
    """
    Predict using the trained model.
    
    Example input:
    {
        "feature1": 1.0,
        "feature2": 0.5,
        ...
    }
    """
    try:
        # Convert input data to a DataFrame for easier handling
        input_df = pd.DataFrame([input_data])
        
        # Convert DataFrame to PyTorch Tensor
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        
        # Perform prediction
        with torch.no_grad():
            logits = model(input_tensor)  # Raw output from the model
            probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()  # Get probabilities
            predicted_class = probabilities.argmax(axis=1).tolist()  # Get predicted class
        
        return {
            "prediction": predicted_class,
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
