from pathlib import Path
import pandas as pd
import typer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.dataframes = {}

    def load_data(self) -> None:
        """Load data from raw data path."""
        # Load the two datasets into memory
        self.dataframes['economic_calendar'] = pd.read_csv(self.data_path / "economic_calendar.csv")
        self.dataframes['exchange_rates'] = pd.read_csv(self.data_path / "usd_chf_prices.csv")

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        print("Proprocessing data...")

        # Example preprocessing: Cleaning and merging the data
        df_calendar = self.dataframes['economic_calendar']
        df_exchange = self.dataframes['exchange_rates']

        # # Flatten MultiIndex columns
        # df_exchange.columns = ['_'.join(col).strip() for col in df_exchange.columns.values]
        
        # # Rename columns
        # df_exchange.columns = ['Price Close', 'Price High', 'Price Low', 'Price Open', 'Volume']
        
        # # Reset the index to make Datetime a regular column
        # df_exchange.reset_index(inplace=True)
        
        # # Rename the "Datetime" column if needed
        # df_exchange.rename(columns={'index': 'Datetime'}, inplace=True)  # Optional if the index name isn't already 'Datetime
        
        df_exchange['timestamp'] = pd.to_datetime(df_exchange['Datetime'])

        df_calendar['time'] = df_calendar['time'].replace('All Day', '00:00')
        df_calendar['time'] = df_calendar['time'].replace('ll DayA', '00:00')
        df_calendar['time'] = df_calendar['time'].fillna('00:00')
        df_calendar['timestamp'] = pd.to_datetime(df_calendar['date'] + ' ' + df_calendar['time'], errors='coerce', dayfirst=True)

        df_exchange = df_exchange.sort_values(by='timestamp')
        df_calendar = df_calendar.sort_values(by='timestamp')

        df_exchange['timestamp'] = df_exchange['timestamp'].dt.tz_localize(None)
        df_calendar['timestamp'] = df_calendar['timestamp'].dt.tz_localize(None)

        print(df_exchange.head())
        print(df_calendar.head())
        
        # Merge the datasets based on the closest previous event timestamp
        merged_data = pd.merge_asof(
            df_exchange,          # Forex data
            df_calendar,         # Economic events data
            on='timestamp',      # Key column to merge on
            direction='backward' # Use the closest preceding event
        )
        # Filter rows where 'event' column is not null
        merged_data = merged_data[merged_data['event'].notnull()]
        merged_data.fillna({'actual': 0, 'forecast': 0, 'previous': 0}, inplace=True)

        # # Lagged Prices: Add previous price values to capture trends.
        merged_data['Close_Lag1'] = merged_data['Price Close'].shift(1)
        merged_data['Close_Lag2'] = merged_data['Price Close'].shift(2)

        # # Price Change: Calculate the price change between consecutive rows.
        merged_data['price_change'] = merged_data['Price Close'] - merged_data['Close_Lag1']

        # # Percentage Change: Calculate the percentage change in price.
        merged_data['price_pct_change'] = (merged_data['price_change'] / merged_data['Close_Lag1']) * 100

        merged_data['actual'] = pd.to_numeric(merged_data['actual'], errors='coerce')
        merged_data['forecast'] = pd.to_numeric(merged_data['forecast'], errors='coerce')
        merged_data['previous'] = pd.to_numeric(merged_data['previous'], errors='coerce')

        merged_data.fillna({'actual': 0, 'forecast': 0, 'previous': 0}, inplace=True)

        # Delta Features: Compute differences between actual and forecast values:
        merged_data['delta_forecast'] = merged_data['actual'] - merged_data['forecast']
        merged_data['delta_previous'] = merged_data['actual'] - merged_data['previous']

        # # # Impact Score: Convert categorical impact values into numerical scores:
        # impact_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        # merged_data['impact_score'] = merged_data['impact'].map(impact_mapping)

        # print(merged_data.head())

        # Get all unique values in the 'importance' column
        # unique_importance_values = merged_data['importance'].unique()
        # print(f"Unique values in 'importance': {unique_importance_values}")

        # Fill missing values and reassign the column
        merged_data['importance'] = merged_data['importance'].fillna('Unknown')

        importance_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 0}
        merged_data['importance_score'] = merged_data['importance'].map(importance_mapping)
        merged_data['Price_Direction'] = (merged_data['price_change'] > 0).astype(int)

        # Save the processed data
        output_folder.mkdir(parents=True, exist_ok=True)
        processed_path = output_folder / "processed_data.csv"
        merged_data.to_csv(processed_path, index=False)
        print(f"Preprocessed data saved to {processed_path}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataframe.get('merged_data', []))

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        if 'merged_data' not in self.dataframes:
            raise ValueError("Data has not been preprocessed yet!")
        return self.dataframes['merged_data'].iloc[index]


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.load_data()
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
