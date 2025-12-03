# Project Configuration Settings

TICKER = "^GSPC"  # S&P 500 Index Ticker Symbol
START_DATE = "2010-01-01"  # Start date for data retrieval
END_DATE = "2023-12-31"  # End date for data retrieval

# Data Settings
TRAIN_TEST_SPLIT = 0.8  # Proportion of data to use for training / testing
WINDOW_SIZE = 60  # Number of days in the input sequence

# Model Settings
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 32  # Size of each training batch

# Paths
DATA_PATH = "data/sp500_data.csv"  # Path to save/load the dataset
MODEL_PATH = "model/sp500_model.keras"  # Path to save/load the trained
SCALER_PATH = "model/sp500_scaler.pkl"  # Path to save/load the scaler