from pathlib import Path

from data import fetch_economic_calendar, fetch_price_data, preprocess

# from fetch_data import fetch_economic_calendar, fetch_price_data
from train import train_model


def main():
    # Define output paths
    calendar_output_path = Path("data/raw/economic_calendar.csv")
    usd_chf_output_path = Path("data/raw/usd_chf_prices.csv")

    # Fetch data
    fetch_economic_calendar(calendar_output_path, "01/12/2024", "31/12/2024")
    fetch_price_data(
        usd_chf_output_path, ticker="USDCHF=X", interval="1h", period="1mo"
    )

    # Define paths
    raw_data_path = Path("data/raw/")  # Path to the raw data directory
    output_folder = Path("data/processed/")  # Path to save the processed data
    # Run preprocessing
    preprocess(raw_data_path, output_folder)

    # # Path to the preprocessed file
    preprocessed_file = Path("data/processed/processed_data.csv")

    # Train a Random Forest model
    # train_model("random_forest", preprocessed_file)

    # Train a Logistic Regression model
    # train_model("logistic_regression", preprocessed_file)

    # Train a Deep Learning model
    train_model(
        "deep_learning", preprocessed_file, input_size=10, hidden_size=64, output_size=2
    )


if __name__ == "__main__":
    main()
