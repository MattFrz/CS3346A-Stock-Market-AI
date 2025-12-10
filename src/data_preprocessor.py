"""
data_preprocessor.py

Loads raw stock data, cleans and normalizes it, and creates sequences for training the model.
Also handles train/test splitting.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add the parent directory to the path so Python can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings from config.py
from config import DATA_PATH, TRAIN_TEST_SPLIT, WINDOW_SIZE, SCALER_PATH, USE_FEATURES

# Import feature engineering
from feature_engineering import add_all_features

def load_data():
    """
    Loads the CSV file saved by data_collector.py.

    Returns:
        pandas.DataFrame: The loaded stock data
    """
    # Check if data.csv exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run data_collector.py first to download the data.")
        return None
    
    # Load data.csv
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows of data")
    return df

def clean_data(df):
    """
    Cleans the data by removing missing values and sorting by date (ascending).

    Args:
        df: Raw dataframe

    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    print(f"Initial shape before data cleaning: {df.shape}")
    print("Cleaning data...")
    
    # Remove any rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing values")
    
    # Sort by date (oldest to newest)
    df = df.sort_index()
    
    # Keep only the columns we need for prediction
    # Using 'Close' as our main target, but keeping others as features
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[columns_to_keep]
    
    print(f"Data cleaned. Final shape: {df.shape}")
    return df

def normalize_data(df):
    """
    Normalizes all features to 0-1 range using MinMaxScaler to make learning more efficient.

    Args:
        df: Cleaned dataframe

    Returns:
        tuple: (normalized_array, scaler_object)
    """
    print("Normalizing data to 0-1 range...")
    
    # Create scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df. values)
    
    # Save the scaler for denormalizing predictions
    scaler_dir = os.path.dirname(SCALER_PATH)
    if scaler_dir:
        os. makedirs(scaler_dir, exist_ok=True)
    
    with open(SCALER_PATH, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print(f"Scaler saved to {SCALER_PATH}")
    
    return scaled_data, scaler

def create_sequences(data, window_size=WINDOW_SIZE, target_column_index=3):
    """
    Creates sequences for model training using sliding window approach.

    Example with window_size = 3:
        Days 1-3 -> predict day 4
        Days 2-4 -> predict day 5
        Days 3-5 -> predict day 6
        etc.

    Args:
        data: Normalized numpy array
        window_size: Number of previous days to use as input
        target_column_index: Index of the column to predict (default 3 = 'Close')

    Returns:
        tuple: (input_sequences, target_prices)
            where input_sequences is the historical data and target_prices is the price to predict
    """
    print(f"Creating sequences with window size {window_size}...")
    
    input_sequences = []  # Historical data
    target_prices = []  # Prices to predict
    
    # Create sliding windows
    for i in range(window_size, len(data)):
        # input_sequences: previous 'window_size' days of all features
        input_sequences.append(data[i - window_size:i])
        
        # target_prices: the 'Close' price of the next day
        target_prices.append(data[i, target_column_index])
    
    # Convert lists to numpy arrays
    input_sequences = np.array(input_sequences)
    target_prices = np.array(target_prices)
    
    print(f"Created {len(input_sequences)} sequences")
    print(f"input_sequences shape: {input_sequences.shape} (samples, timesteps, features)")
    print(f"target_prices shape: {target_prices.shape} (samples,)")
    
    return input_sequences, target_prices

def split_train_test(input_sequences, target_prices, split_ratio=TRAIN_TEST_SPLIT):
    """
    Splits data into training and testing sets.
    We don't shuffle because we want older data for training and newer data for testing.

    Args:
        input_sequences: Historical data sequences
        target_prices: Prices to predict
        split_ratio: Proportion to use for training (e.g. 0.8 = 80% towards training)
    
    Returns:
        tuple: (sequences_train, sequences_test, prices_train, prices_test)
    """
    print(f"Splitting data: {split_ratio*100:.0f}% train, {(1-split_ratio)*100:.0f}% test...")
    
    # Calculate split index
    split_index = int(len(input_sequences) * split_ratio)
    
    # Split the data
    sequences_train = input_sequences[:split_index]
    sequences_test = input_sequences[split_index:]
    prices_train = target_prices[:split_index]
    prices_test = target_prices[split_index:]
    
    print(f"Training samples: {len(sequences_train)}")
    print(f"Testing samples: {len(sequences_test)}")
    
    return sequences_train, sequences_test, prices_train, prices_test

def preprocess_pipeline(use_features=USE_FEATURES):
    """
    Runs the complete preprocessing pipeline:
    1. Load data
    2. Clean data
    3. Add features (if enabled)
    4. Normalize data
    5. Create sequences
    6. Split train/test

    Args:
        use_features:
            If True, adds technical indicators.
            If False, uses only OHLCV.
            Defaults to USE_FEATURES from config.py

    Returns:
        tuple: (sequences_train, sequences_test, prices_train, prices_test, scaler)
    """
    print("\n" + "="*50)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*50 + "\n")
    
    # Step 1: Load
    df = load_data()
    if df is None:
        return None
    
    # Step 2: Clean
    df_cleaned = clean_data(df)
    
    # Step 3: Add Features
    if use_features:
        df_processed = add_all_features(df_cleaned)
        # Update target column index (find 'Close' column position)
        target_col_idx = list(df_processed.columns).index('Close')
    else:
        print("\nSkipping feature engineering (using raw OHLCV only)")
        df_processed = df_cleaned
        target_col_idx = 3  # 'Close' is at index 3 in OHLCV
    
    # Step 4: Normalize
    scaled_data, scaler = normalize_data(df_processed)
    
    # Step 5: Create sequences
    input_sequences, target_prices = create_sequences(scaled_data, target_column_index=target_col_idx)
    
    # Step 6: Split
    sequences_train, sequences_test, prices_train, prices_test = split_train_test(input_sequences, target_prices)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50 + "\n")
    
    return sequences_train, sequences_test, prices_train, prices_test, scaler

# Run the pipeline if data_preprocessor.py is executed directly
if __name__ == "__main__":
    # Uses USE_FEATURES from config.py by default
    result = preprocess_pipeline()
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()

    if result is not None:
        sequences_train, sequences_test, prices_train, prices_test, scaler = result
        
        print("\nFinal shapes:")
        print(f"sequences_train: {sequences_train. shape}")
        print(f"sequences_test: {sequences_test.shape}")
        print(f"prices_train: {prices_train.shape}")
        print(f"prices_test: {prices_test.shape}")
        print("\nData ready for model training")
