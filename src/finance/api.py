from fastapi import FastAPI, HTTPException
import torch
import pandas as pd

# Load your trained model
MODEL_PATH = "model/model.pth"

try:
    model = torch.load(MODEL_PATH)
    model.eval()  # Set the model to evaluation mode
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
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
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Convert DataFrame to Tensor
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        
        # Perform the prediction
        with torch.no_grad():
            prediction = model(input_tensor).numpy().tolist()
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
