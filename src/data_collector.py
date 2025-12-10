"""
data_collector.py

Downloads historical stock data for a specified ticker symbol and date
 range using yfinance and saves it to a CSV file in the data/ folder.

"""

import sys
import os
import yfinance as yf
import pandas as pd

# Add the parent directory to the path so Python can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings from config.py
from config import TICKER, START_DATE, END_DATE, DATA_PATH

def download_sp500_data():
    """"
    Downloads historical data from Yahoo Finance and saves it to a CSV file.
    """
    print(f"Downloading data for {TICKER} from {START_DATE} to {END_DATE}...")

    # Use yfinance to download daily OHLCV data
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",   # daily data
        auto_adjust=False
    )

    if df is None or df.empty:
        print("No data was downloaded. Check your ticker or date range.")
        return None

    # Make sure the data folder exists
    data_dir = os.path.dirname(DATA_PATH)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    
    # Flatten column names if they're multi-level (removes ticker symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Save to CSV
    df.to_csv(DATA_PATH)
    print(f"Data saved to {DATA_PATH}")

    return df

# Download sp500 data if data_collector.py is executed directly
if __name__ == "__main__":
    download_sp500_data()