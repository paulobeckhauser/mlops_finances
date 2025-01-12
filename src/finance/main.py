import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from data import fetch_economic_calendar, fetch_price_data, preprocess
# from fetch_data import fetch_economic_calendar, fetch_price_data
from train import train_model
from evaluate import evaluate_model
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
import logging
import sys

# Redirect stdout to the logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

@hydra.main(version_base=None, config_path="../../configs", config_name="config")

def main(cfg: DictConfig):
    log.info("Hydra is working!")
    print("This print statement will now be captured in logs.")
    # Print the loaded configuration for debugging
    print("Loaded Configuration:")
    print(OmegaConf.to_yaml(cfg))
    # Access nested keys under cfg.model
    print(f"Training {cfg.model_name} model...")
    print(f"Learning rate: {cfg.model.lr}")
    print(f"Epochs: {cfg.model.epochs}")

    # Define output paths
    calendar_output_path = Path("data/raw/economic_calendar.csv")
    usd_chf_output_path = Path("data/raw/usd_chf_prices.csv")

    # Fetch data
    # fetch_economic_calendar(calendar_output_path, '01/01/2023', '31/12/2024')
    # fetch_price_data(usd_chf_output_path, ticker='USDCHF=X', interval='1h', period='2y')
    # Define paths
    raw_data_path = Path("data/raw/")  # Path to the raw data directory
    output_folder = Path("data/processed/")  # Path to save the processed data
    # Run preprocessing
    preprocess(raw_data_path, output_folder)

    # # Path to the preprocessed file
    preprocessed_file = Path("data/processed/processed_data.csv")

    checkpoint_path = Path("checkpoints/best-checkpoint.ckpt")

    # Train the model
    print(f"Training {cfg.model_name} model...")
    train_model(
        cfg.model_name,  # Positional argument
        preprocessed_file,
        **cfg.model  # Unpack model-specific parameters only
    )

    # Evaluate the model
    print(f"Evaluating {cfg.model_name} model...")
    evaluate_model(
        checkpoint_path,
        preprocessed_file,
        batch_size=cfg.model.get("batch_size", 32)  # Default to 32 if not defined
    )

if __name__ == "__main__":
    main()