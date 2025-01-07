from pathlib import Path

from data import fetch_economic_calendar, fetch_price_data, preprocess

# from fetch_data import fetch_economic_calendar, fetch_price_data
from train import train_model
from evaluate import evaluate_model


def main():
    # Define output paths
    # calendar_output_path = Path("data/raw/economic_calendar.csv")
    # usd_chf_output_path = Path("data/raw/usd_chf_prices.csv")

    # # Fetch data
    # fetch_economic_calendar(calendar_output_path, '01/01/2023', '31/12/2024')
    # fetch_price_data(usd_chf_output_path, ticker='USDCHF=X', interval='1h', period='1yo')
    # fetch_price_data(usd_chf_output_path, ticker='USDCHF=X', interval='1h', period='2y')
    # Define paths
    # raw_data_path = Path("data/raw/")  # Path to the raw data directory
    # output_folder = Path("data/processed/")  # Path to save the processed data
    # # Run preprocessing
    # preprocess(raw_data_path, output_folder)

    # # Path to the preprocessed file
    preprocessed_file = Path("data/processed/processed_data.csv")

    # # Train a Random Forest model
    # train_model("random_forest", preprocessed_file)

    # # Train a Logistic Regression model
    # train_model("logistic_regression", preprocessed_file)

    # Train a Deep Learning model
    # train_model("deep_learning", preprocessed_file, input_size=10, hidden_size=64, output_size=2)

    model_name = "deep_learning"
    model_params = {
        "epochs": 20,
        "lr": 0.0005,
        "num_classes": 3,
    }

    train_model(model_name, preprocessed_file, **model_params)


    # Evaluate the model
    checkpoint_path = Path("checkpoints/best-checkpoint.ckpt")  # Path to the saved checkpoint
    print(f"Evaluating {model_name} model...")
    evaluate_model(checkpoint_path, preprocessed_file, batch_size=32)

if __name__ == "__main__":
    main()
