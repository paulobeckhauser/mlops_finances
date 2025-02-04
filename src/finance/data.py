# mypy: disallow-untyped-defs
from pathlib import Path
from typing import Any
import requests

import investpy
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.dataframes: dict[str, pd.DataFrame] = {}

    def load_data(self) -> None:
        """Load data from raw data path."""
        # Load the two datasets into memory
        self.dataframes["economic_calendar"] = pd.read_csv(
            self.data_path / "economic_calendar.csv"
        )
        self.dataframes["exchange_rates"] = pd.read_csv(
            self.data_path / "usd_chf_prices.csv"
        )

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # print("Proprocessing data...")

        # Example preprocessing: Cleaning and merging the data
        df_calendar = self.dataframes["economic_calendar"]
        df_exchange = self.dataframes["exchange_rates"]

        df_exchange["timestamp"] = pd.to_datetime(df_exchange["Datetime"])

        # pd.set_option('display.max_columns', None)  # Show all columns

        # print(df_exchange.head(10))

        # print(df_calendar.shape)
        # df_calendar = df_calendar[(~df_calendar['actual'].isna()) &
        #                              (df_calendar['currency'] == 'USD') &
        #                              (df_calendar['importance'] == 'high')].copy()
        # print(df_calendar.shape)
        # df_calendar_show = df_calendar[(df_calendar['currency'] == 'USD')]
        # print(df_calendar.head())
        # print(df_calendar_show.shape)
        # print(df_calendar_show['event'].unique())

        # 1st Decision ->

        # df_calendar['time'] = df_calendar['time'].replace('All Day', '00:00')
        # df_calendar['time'] = df_calendar['time'].replace('ll DayA', '00:00')
        # df_calendar['time'] = df_calendar['time'].fillna('00:00')
        # df_calendar['timestamp'] = pd.to_datetime(df_calendar['date'] + ' ' + df_calendar['time'], errors='coerce', dayfirst=True)
        df_calendar["timestamp"] = pd.to_datetime(
            df_calendar["date"] + " " + df_calendar["time"],
            format="%d/%m/%Y %H:%M",  # Adjust the format to match your data
            errors="coerce",
            dayfirst=True,  # Since day is the first part of the date
        )
        # print(df_calendar.shape)

        df_calendar["time"] = df_calendar["time"].replace("All Day", "00:00")
        df_calendar["time"] = df_calendar["time"].replace("ll DayA", "00:00")
        df_calendar["time"] = df_calendar["time"].fillna("00:00")
        df_calendar["timestamp"] = pd.to_datetime(
            df_calendar["date"] + " " + df_calendar["time"],
            errors="coerce",
            dayfirst=True,
        )

        df_exchange["timestamp"] = df_exchange["timestamp"].dt.tz_localize(None)
        df_calendar["timestamp"] = df_calendar["timestamp"].dt.tz_localize(None)

        # df_calendar_test = df_calendar[df_calendar['event'] == 'Initial Jobless Claims']
        # print(df_calendar_test)
        pd.set_option("display.max_columns", None)  # Show all columns
        # print(df_exchange.head())
        # print(df_calendar.head())

        # Check for null values in the timestamp columns
        # print("Null values in df_exchange['timestamp']:", df_exchange['timestamp'].isna().sum())
        # print("Null values in df_calendar['timestamp']:", df_calendar['timestamp'].isna().sum())

        # Drop rows with null timestamps
        df_exchange = df_exchange.dropna(subset=["timestamp"])
        df_calendar = df_calendar.dropna(subset=["timestamp"])

        # Sort DataFrames by the timestamp column
        df_exchange = df_exchange.sort_values(by="timestamp").reset_index(drop=True)
        df_calendar = df_calendar.sort_values(by="timestamp").reset_index(drop=True)

        # print("Sample of df_exchange['timestamp']:", df_exchange['timestamp'].head())
        # print("Sample of df_calendar['timestamp']:", df_calendar['timestamp'].head())

        # # # Merge the datasets based on the closest previous event timestamp
        merged_data = pd.merge_asof(
            df_exchange,  # Forex data
            df_calendar,  # Economic events data
            on="timestamp",  # Key column to merge on
            direction="backward",  # Use the closest preceding event
        )

        # merged_data_filter = merged_data[(merged_data['timestamp'] >= '2024-12-26 00:00:00') & (merged_data['timestamp'] < '2024-12-27 00:00:00')]
        # print(merged_data_filter)
        # print(merged_data_filter.shape)
        # merget_data = pd.merge()

        # print(merged_data.head())
        # print(merged_data.shape)
        # # Filter rows where 'event' column is not null
        merged_data = merged_data[merged_data["event"].notnull()]
        merged_data.fillna({"actual": 0, "forecast": 0, "previous": 0}, inplace=True)

        # # # # Lagged Prices: Add previous price values to capture trends.
        merged_data["Close_Lag1"] = merged_data["Price Close"].shift(1)
        merged_data["Close_Lag2"] = merged_data["Price Close"].shift(2)

        # # # # Price Change: Calculate the price change between consecutive rows.
        merged_data["price_change"] = (
            merged_data["Price Close"] - merged_data["Close_Lag1"]
        )

        # # # # Percentage Change: Calculate the percentage change in price.
        merged_data["price_pct_change"] = (
            merged_data["price_change"] / merged_data["Close_Lag1"]
        ) * 100

        merged_data["actual"] = pd.to_numeric(merged_data["actual"], errors="coerce")
        merged_data["forecast"] = pd.to_numeric(
            merged_data["forecast"], errors="coerce"
        )
        merged_data["previous"] = pd.to_numeric(
            merged_data["previous"], errors="coerce"
        )

        merged_data.fillna({"actual": 0, "forecast": 0, "previous": 0}, inplace=True)

        # Delta Features: Compute differences between actual and forecast values:
        merged_data["delta_forecast"] = merged_data["actual"] - merged_data["forecast"]
        merged_data["delta_previous"] = merged_data["actual"] - merged_data["previous"]

        # # # Impact Score: Convert categorical impact values into numerical scores:
        # impact_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        # merged_data['impact_score'] = merged_data['impact'].map(impact_mapping)

        # print(merged_data.head())

        # # Get all unique values in the 'importance' column
        # # unique_importance_values = merged_data['importance'].unique()
        # # print(f"Unique values in 'importance': {unique_importance_values}")

        # Fill missing values and reassign the column
        merged_data["importance"] = merged_data["importance"].fillna("Unknown")

        importance_mapping = {"Low": 1, "Medium": 2, "High": 3, "Unknown": 0}
        merged_data["importance_score"] = merged_data["importance"].map(
            importance_mapping
        )
        merged_data["Price_Direction"] = (merged_data["price_change"] > 0).astype(int)

        # Save the processed data
        output_folder.mkdir(parents=True, exist_ok=True)
        processed_path = output_folder / "processed_data.csv"
        merged_data.to_csv(processed_path, index=False)
        print(f"Preprocessed data saved to {processed_path}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframes.get("merged_data", []))

    def __getitem__(self, index: int) -> Any:
        """Return a given sample from the dataset."""
        if "merged_data" not in self.dataframes:
            raise ValueError("Data has not been preprocessed yet!")
        return self.dataframes["merged_data"].iloc[index]


def fetch_economic_calendar(output_path: Path, start_date: str, end_date: str) -> None:
    """Fetch raw economic data and save it to the specified path.
    Right now, we are getting time zone GMT+01:00
    """
    # print("Fetching raw economic calendar data...")
    calendar_data = investpy.economic_calendar(
        time_zone="GMT",  # Set your desired time zone
        from_date=start_date,
        to_date=end_date,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_data.to_csv(output_path, index=False)

    # pd.set_option('display.max_columns', None)  # Show all columns
    # print(calendar_data.columns)
    # filtered_df = calendar_data[(calendar_data['zone'] == 'brazil')]
    # print(filtered_df.head())
    # print(f"Data saved to {output_path}")


def fetch_price_data(
    output_path: Path, ticker: str, interval: str = "1h", period: str = "1mo"
) -> None:
    """
    Fetch historical price data for a given ticker and save it to the specified path.

    Parameters:
    - ticker: str, e.g., 'USDCHF=X' for USD/CHF currency pair.
    - interval: str, e.g., '1h', '15m'.
    - period: str, e.g., '1mo', '5d'.

    Right now the timezone is in UTC
    """
    # print(f"Feting price data for {ticker}...")
    data = yf.download(ticker, interval=interval, period=period)
    # Flatten MultiIndex columns
    data.columns = ["_".join(col).strip() for col in data.columns.values]

    # Rename columns
    data.columns = ["Price Close", "Price High", "Price Low", "Price Open", "Volume"]

    # Reset the index to make Datetime a regular column
    data.reset_index(inplace=True)

    # Rename the "Datetime" column if needed
    data.rename(
        columns={"index": "Datetime"}, inplace=True
    )  # Optional if the index name isn't already 'Datetime

    if not data.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        # print(f"Price data for {ticker} saved to {output_path}")
    else:
        print(f"No data found for {ticker}. Please check the ticker or parameters.")

    # pd.set_option('display.max_columns', None)  # Show all columns
    # print(data.columns)
    # print(data.shape)
    # # print(data.info())
    # filtered_df = data[(data['Volume'] == 0)]
    # print(filtered_df.head())
    # print(filtered_df.shape)
    # unique_filtered = data['Volume'].unique()
    # print(unique_filtered)


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    # print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.load_data()
    dataset.preprocess(output_folder)


def load_preprocessed_data(
    preprocessed_file: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the preprocessed data and prepare it for training and evaluation.

    Args:
        preprocessed_file (Path): Path to the preprocessed CSV file.

    Returns:
        pd.DataFrame: Processed feature set X.
        pd.Series: Target variable y.
    """
    print(f"Loading preprocessed data from {preprocessed_file}...")
    merged_data = pd.read_csv(preprocessed_file)

    # Drop rows with NaN values in the specified columns
    X = merged_data[
        ["Close_Lag1", "Close_Lag2", "importance_score", "delta_forecast"]
    ].dropna()

    # Align `y` with `X`'s index
    print("Aligning target variable y with X's index...")
    y = merged_data.loc[X.index, "Price_Direction"]

    print("Data successfully loaded and processed!")
    return X, y


def get_training_data(
    preprocessed_file: Path, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the preprocessed data, prepare features and labels, and split into training and
    testing sets.

    Args:
        preprocessed_file (Path): Path to the preprocessed CSV file.
        test_size (float): Fraction of the data to use for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # Load and prepare data
    X, y = load_preprocessed_data(preprocessed_file)

    # Split into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def download_from_public_url(public_url: str, destination_file_name: str) -> None:
    """
    Download a file from a public URL.
    Args:
        public_url (str): The public URL of the file.
        destination_file_name (str): The local path to save the downloaded file.
    """
    response = requests.get(public_url, stream=True)
    if response.status_code == 200:
        # Ensure the destination directory exists
        Path(destination_file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(destination_file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {public_url} to {destination_file_name}.")
    else:
        raise Exception(f"Failed to download file. HTTP Status: {response.status_code}")


if __name__ == "__main__":
    # Define output paths
    calendar_output_path = Path("data/raw/economic_calendar.csv")
    usd_chf_output_path = Path("data/raw/usd_chf_prices.csv")

    # Fetch data
    # fetch_economic_calendar(calendar_output_path, '01/12/2024', '31/12/2024')
    # fetch_price_data(usd_chf_output_path, ticker='USDCHF=X', interval='1h', period='1mo')
    # import typer
    # typer.run(preprocess)
