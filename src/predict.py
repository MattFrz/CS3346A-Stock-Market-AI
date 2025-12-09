"""
predict.py

Uses the trained model to predict the next day's stock price.
Loads the most recent data, processes it, and makes a prediction.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

# Add the parent directory to the path to find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings from config.py
from config import DATA_PATH, MODEL_PATH, SCALER_PATH, WINDOW_SIZE, USE_FEATURES

# Import feature engineering
from feature_engineering import add_all_features


def load_trained_model():
    """
    Loads the trained model from storage.
    
    Returns:
        keras.Model: The loaded trained model, or None if the model was not found
    """
    # Check if sp500_model.keras exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model")
        return None
    
    # Load sp500_model.keras
    print(f"Loading trained model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
    return model


def load_scaler():
    """
    Loads the scaler used during training.
    The scaler is used to normalize the data consistently.
    
    Returns:
        MinMaxScaler: The loaded scaler, or None if the scaler was not found
    """
    # Check if sp500_scaler.pkl exists
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler not found at {SCALER_PATH}")
        print("Please run data_preprocessor.py first to get the scaler")
        return None
    
    # Load sp500_scaler.pkl
    print(f"Loading scaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully")
    return scaler


def get_latest_data(num_days=WINDOW_SIZE, use_features=USE_FEATURES):
    """
    Gets the most recent stock data for prediction.
    
    Args:
        num_days: Number of days to retrieve (use extra for feature calculations)
        use_features: Whether to add technical indicators
        
    Returns:
        tuple: (processed_dataframe, target_column_name) or (None, None) if data file not found
    """
    # Check if sp500_data.csv exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run data_collector.py first to get data")
        return None, None
    
    # Load sp500_data.csv
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Sort by date (oldest to newest)
    df = df.sort_index()
    
    # Keep only the columns we need for prediction
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[columns_to_keep]
    
    # Get extra data for feature calculations
    if use_features:
        print("Adding technical indicators...")
        extra_rows = 60
        df_extended = df.tail(num_days + extra_rows)    # Get data from the past (num_days + extra_rows) days
        
        # Add features
        df_with_features = add_all_features(df_extended)
        
        # Trim data back to the past 'num_days' days
        latest_data = df_with_features.tail(num_days)
        target_column = 'Close'
        
        print(f"Retrieved data with {len(latest_data.columns)} features")
    else:
        print("Using raw OHLCV data (no features)...")
        latest_data = df.tail(num_days) # Get data from the past 'num_days' days
        target_column = 'Close'
        
        print(f"Retrieved data with 5 basic features")
    
    print(f"Date range: {latest_data.index[0].date()} to {latest_data.index[-1].date()}")
    
    return latest_data, target_column


def prepare_prediction_data(data, scaler, target_column):
    """
    Prepares the data for prediction by normalizing it.
    
    Args:
        data: Raw dataframe with recent stock data
        scaler: The MinMaxScaler used during training
        target_column: Name of the column to predict
    
    Returns:
        tuple: (prediction_input, target_column_index)
            prediction_input has shape (1, window_size, features)
    """
    print("Preparing data for prediction...")
    
    # Find the index of the target column
    target_column_index = list(data.columns).index(target_column)
    
    # Normalize the data using the same scaler from training
    normalized_data = scaler.transform(data.values)
    
    # Reshape for model input: (1 sample, window_size, num_features)
    prediction_input = normalized_data.reshape(1, normalized_data.shape[0], normalized_data.shape[1])
    
    print(f"Data prepared, shape: {prediction_input.shape}")
    print(f"Target column '{target_column}' at index {target_column_index}")
    
    return prediction_input, target_column_index


def make_prediction(model, prediction_input, scaler, target_column_index):
    """
    Makes a prediction using the trained model.
    
    Args:
        model: Trained keras model
        prediction_input: Prepared input data
        scaler: MinMaxScaler to denormalize the prediction
        target_column_index: Index of the target column in the scaler
    
    Returns:
        float: Predicted closing price
    """
    print("Making prediction...")
    
    # Get prediction from model (normalized to 0-1 range)
    normalized_prediction = model.predict(prediction_input, verbose=0)

    # Create dummy array with the shape of the original data
    n_features = scaler.n_features_in_
    dummy_array = np.zeros((1, n_features))
    dummy_array[0, target_column_index] = normalized_prediction[0, 0]   # Put prediction in target column
    
    # Denormalize to get real price
    denormalized = scaler.inverse_transform(dummy_array)
    predicted_price = denormalized[0, target_column_index]
    
    print("Prediction complete")
    return predicted_price


def predict_next_day():
    """
    Runs the complete pipeline to predict tomorrow's stock price:
    1. Load trained model
    2. Load scaler
    3. Get latest data
    4. Prepare data for prediction
    5. Make Prediction
    6. Get today's price for comparison
    
    Returns:
        dict: Dictionary with prediction results, or None if error
    """
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION")
    print("="*60 + "\n")
    
    # Step 1: Load the trained model
    model = load_trained_model()
    if model is None:
        return None
    
    # Step 2: Load the scaler
    scaler = load_scaler()
    if scaler is None:
        return None
    
    # Step 3: Get latest data
    latest_data, target_column = get_latest_data()
    if latest_data is None:
        return None
    
    # Step 4: Prepare data for prediction
    prediction_input, target_column_index = prepare_prediction_data(latest_data, scaler, target_column)
    
    # Step 5: Make prediction
    predicted_price = make_prediction(model, prediction_input, scaler, target_column_index)
    
    # Step 6: Get today's price for comparison
    todays_close = latest_data[target_column].iloc[-1]
    price_change = predicted_price - todays_close
    percent_change = (price_change / todays_close) * 100
    
    # Prepare results as dict
    results = {
        'current_price': todays_close,
        'predicted_price': predicted_price,
        'price_change': price_change,
        'percent_change': percent_change,
        'current_date': latest_data.index[-1].date(),
        'features_used': USE_FEATURES
    }
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Features Used:     {'Technical Indicators' if USE_FEATURES else 'Raw OHLCV Only'}")
    print(f"Current Date:      {results['current_date']}")
    print(f"Current Price:     ${results['current_price']:,.2f}")
    print(f"Predicted Price:   ${results['predicted_price']:,.2f}")
    print(f"Price Change:      ${results['price_change']:+,.2f}")
    print(f"Percent Change:    {results['percent_change']:+.2f}%")
    
    if price_change > 0:
        print(f"\nPrediction: UP")
    elif price_change < 0:
        print(f"\nPrediction: DOWN")
    else:
        print(f"\nPrediction: FLAT")
    
    print("="*60 + "\n")
    
    return results

# Run prediction pipeline if this file is executed directly.
if __name__ == "__main__":
    predict_next_day()