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

    # # Access the model name directly
    # # print(f"Training {cfg.model_name} model...")
    # print(f"Training {cfg.model_name} model...")

    # # Example: Access specific configuration values
    # print(f"Learning rate: {cfg.lr}")
    # print(f"Epochs: {cfg.epochs}")
    # # Debug Hydra configuration paths
    # hydra_cfg = HydraConfig.get()
    # print("Hydra Search Path:")
    # print("\n".join(hydra_cfg.hydra.runtime.config_search_path))
    # print("Loaded Configuration:")
    # print(cfg)


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

    checkpoint_path = Path("checkpoints/best-checkpoint.ckpt")

    # # Train a Random Forest model
    # train_model("random_forest", preprocessed_file)

    # # Train a Logistic Regression model
    # train_model("logistic_regression", preprocessed_file)

    # Train a Deep Learning model
    # train_model("deep_learning", preprocessed_file, input_size=10, hidden_size=64, output_size=2)

    # model_name = "deep_learning"
    # model_params = {
    #     "epochs": 20,
    #     "lr": 0.0005,
    #     "num_classes": 3,
    # }

    # Train the model
    print(f"Training {cfg.model_name} model...")
    train_model(
        cfg.model_name,  # Positional argument
        preprocessed_file,
        **cfg.model  # Unpack model-specific parameters only
    )

    # # Train the specified model
    # train_model(
    #     args.model_name,
    #     preprocessed_file,
    #     epochs=args.epochs,
    #     lr=args.lr,
    #     num_classes=args.num_classes,
    #     input_size=args.input_size,
    #     hidden_size=args.hidden_size,
    #     output_size=args.output_size
    # )

    # train_model(model_name, preprocessed_file, **model_params)


    # Evaluate the model
    # checkpoint_path = Path("checkpoints/best-checkpoint.ckpt")  # Path to the saved checkpoint
    # print(f"Evaluating {model_name} model...")
    # Evaluate the model
    print(f"Evaluating {cfg.model_name} model...")
    evaluate_model(
        checkpoint_path,
        preprocessed_file,
        batch_size=cfg.model.get("batch_size", 32)  # Default to 32 if not defined
    )

if __name__ == "__main__":
    main()
    ## to run the script:
    ## python src/finance/main.py --model_name deep_learning --epochs 30 --lr 0.001 --num_classes 4 --batch_size 16 --hidden_size 128
