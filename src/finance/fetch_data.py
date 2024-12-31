from pathlib import Path
import investpy
import yfinance as yf

def fetch_economic_calendar(output_path: Path, start_date:str, end_date: str):
    """Fetch raw economic data and save it to the specified path."""
    print("Fetching raw economic calendar data...")
    calendar_data = investpy.economic_calendar(
        from_date=start_date,
        to_date=end_date
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def fetch_price_data(output_path: Path, ticker: str, interval: str = '1h', period: str = '1mo'):
    """
    Fetch historical price data for a given ticker and save it to the specified path.
    
    Parameters:
    - ticker: str, e.g., 'USDCHF=X' for USD/CHF currency pair.
    - interval: str, e.g., '1h', '15m'.
    - period: str, e.g., '1mo', '5d'.
    """
    print(f"Feting price data for {ticker}...")
    data = yf.download(ticker, interval=interval, period=period)
    if not data.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"Price data for {ticker} saved to {output_path}")
    else:
        print(f"No data found for {ticker}. Please check the ticker or parameters.")

if __name__ == "__main__":
    # Define output paths
    calendar_output_path = Path("data/raw/economic_calendar.csv")
    usd_chf_output_path = Path("data/raw/usd_chf_prices.csv")

    # Fetch data
    fetch_economic_calendar(calendar_output_path, '01/12/2024', '31/12/2024')
    fetch_price_data(usd_chf_output_path, ticker='USDCHF=X', interval='1h', period='1mo')
    