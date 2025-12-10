"""
feature_engineering.py

Adds technical indicators (features) to raw stock data to improve model predictions.
This runs AFTER cleaning but BEFORE normalization.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys. path.append(os.path. dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_moving_averages(df):
    """
    Calculate Simple Moving Averages (SMA) for different time windows.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added MA columns
    """
    print("Calculating Moving Averages...")
    
    df['MA_7'] = df['Close'].rolling(window=7).mean()   # 1 week
    df['MA_21'] = df['Close'].rolling(window=21).mean()  # ~1 month
    df['MA_50'] = df['Close']. rolling(window=50).mean()  # ~quarter
    
    # Calculate ratios (how current price compares to moving averages)
    df['Price_to_MA7'] = df['Close'] / df['MA_7']
    df['Price_to_MA21'] = df['Close'] / df['MA_21']
    
    return df

def calculate_rsi(df, period=14):
    """
    Calculate Relative Strength Index (RSI) - momentum indicator.
    
    RSI ranges from 0-100:
    - Below 30: Oversold (might go up)
    - Above 70: Overbought (might go down)
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default 14 days)
    
    Returns:
        DataFrame with RSI column
    """
    print(f"Calculating RSI with period {period}...")
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta. where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period). mean()
    avg_loss = loss.rolling(window=period). mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) - trend indicator.
    
    MACD shows relationship between two moving averages:
    - Positive MACD: Uptrend
    - Negative MACD: Downtrend
    - MACD crossing signal line: Potential trend change
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        DataFrame with MACD columns
    """
    print(f"Calculating MACD ({fast}, {slow}, {signal})...")
    
    # Calculate Exponential Moving Averages
    ema_fast = df['Close']. ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close']. ewm(span=slow, adjust=False).mean()
    
    # MACD line
    df['MACD'] = ema_fast - ema_slow
    
    # Signal line
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    
    # MACD histogram (difference between MACD and signal)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands - volatility indicator.
    
    Shows if price is "high" or "low" compared to recent average:
    - Price near upper band: Might be overbought
    - Price near lower band: Might be oversold
    
    Args:
        df: DataFrame with OHLCV data
        period: Moving average period (default 20)
        std_dev: Number of standard deviations (default 2)
    
    Returns:
        DataFrame with Bollinger Band columns
    """
    print(f"Calculating Bollinger Bands (period={period}, std={std_dev})...")
    
    # Middle band (SMA)
    df['BB_Middle'] = df['Close']. rolling(window=period).mean()
    
    # Standard deviation
    rolling_std = df['Close'].rolling(window=period).std()
    
    # Upper and lower bands
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    
    # Calculate band width (volatility measure)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Calculate %B (where price is within bands)
    # 0 = at lower band, 1 = at upper band
    df['BB_PercentB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df

def calculate_volatility(df, period=14):
    """
    Calculate price volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
    
    Returns:
        DataFrame with volatility column
    """
    print(f"Calculating volatility (period={period})...")
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate rolling volatility
    df['Volatility'] = df['Daily_Return'].rolling(window=period). std()
    
    return df

def calculate_volume_features(df, period=20):
    """
    Calculate volume-based features.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
    
    Returns:
        DataFrame with volume features
    """
    print(f"Calculating volume features (period={period})...")
    
    # Average volume
    df['Volume_MA'] = df['Volume'].rolling(window=period).mean()
    
    # Volume ratio (current vs average)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

def add_all_features(df):
    """
    Adds ALL technical indicators to the dataframe.
    
    Args:
        df: Cleaned DataFrame with OHLCV data
    
    Returns:
        DataFrame with all features added
    """
    print("\n" + "="*50)
    print("STARTING FEATURE ENGINEERING")
    print("="*50 + "\n")
    
    print(f"Initial shape: {df.shape}")
    
    # Make a copy to avoid modifying original
    df_features = df.copy()
    
    # Add all indicators
    df_features = calculate_moving_averages(df_features)
    df_features = calculate_rsi(df_features)
    df_features = calculate_macd(df_features)
    df_features = calculate_bollinger_bands(df_features)
    df_features = calculate_volatility(df_features)
    df_features = calculate_volume_features(df_features)
    
    # Drop rows with NaN values (from rolling calculations)
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    removed_rows = initial_rows - len(df_features)
    
    print(f"\nRemoved {removed_rows} rows due to indicator calculations")
    print(f"Final shape: {df_features.shape}")
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*50 + "\n")
    
    # Show what features we have
    print("Features created:")
    for i, col in enumerate(df_features.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df_features

# Test feature engineering on its own when executed directly
if __name__ == "__main__":
    from data_preprocessor import load_data, clean_data
    
    print("Testing feature engineering module.. .\n")
    
    # Load and clean data
    df = load_data()
    if df is not None:
        df_cleaned = clean_data(df)
        
        # Add features
        df_with_features = add_all_features(df_cleaned)
        
        print("\nSample of data with features:")
        print(df_with_features.tail())
        
        print(f"\nOriginal columns: {len(df_cleaned.columns)}")
        print(f"Columns after feature engineering: {len(df_with_features.columns)}")
        print(f"New features added: {len(df_with_features.columns) - len(df_cleaned.columns)}")