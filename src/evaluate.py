"""
evaluate.py

Tests the trained model's accuracy on the test dataset.
Also calculates and interprets performance metrics (RMSE, MAE, R², MAPE, Direction Accuracy).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add the parent directory to the path so Python can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import settings from config.py
import config 

def make_predictions(model, X_test, y_test, scaler):
    """
    Make predictions on test data and denormalize values.
    
    Args:
        model: Trained model
        X_test: Test input sequences
        y_test: Actual test values (normalized)
        scaler: Scaler for denormalization
    
    Returns:
        y_test_actual: Actual prices (denormalized)
        y_pred_actual: Predicted prices (denormalized)
    """
    print("\nMaking predictions on test data...")
    
    # Make predictions (normalized)
    y_pred = model.predict(X_test, verbose=0)
    
    # Denormalize predictions and actual values
    # Assuming the close price is the first feature (index 0)
    y_test_actual = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), 
                       np.zeros((len(y_test), scaler.n_features_in_ - 1))], axis=1)
    )[:, 0]
    
    y_pred_actual = scaler.inverse_transform(
        np.concatenate([y_pred,
                       np.zeros((len(y_pred), scaler.n_features_in_ - 1))], axis=1)
    )[:, 0]
    
    print(f"Predictions complete: {len(y_pred_actual)} values")
    
    return y_test_actual, y_pred_actual

def calculate_metrics(y_actual, y_pred):
    """
    Calculate evaluation metrics.
    """
    print("\nCalculating evaluation metrics...")
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_actual, y_pred)
    
    # R² Score
    r2 = r2_score(y_actual, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    # Direction Accuracy (did we predict up/down correctly?)
    actual_direction = np.diff(y_actual) > 0
    pred_direction = np.diff(y_pred) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }
    
    return metrics

def print_metrics(metrics):
    """
    Print evaluation metrics in a formatted way.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    print(f"Root Mean Squared Error (RMSE): ${metrics['RMSE']:.2f}")
    print(f"Mean Absolute Error (MAE):      ${metrics['MAE']:.2f}")
    print(f"R² Score (Accuracy):             {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    print(f"Mean Absolute % Error (MAPE):    {metrics['MAPE']:.2f}%")
    print(f"Direction Accuracy:              {metrics['Direction_Accuracy']:.2f}%")
    print("="*60)
    
    # Interpret metrics
    print("\nINTERPRETATION:")

    # Interpret RMSE
    if metrics['RMSE'] < 50:
        print(" Excellent: Predictions are very accurate (RMSE < $50)")
    elif metrics['RMSE'] < 100:
        print(" Good: Predictions are reasonably accurate (RMSE < $100)")
    elif metrics['RMSE'] < 200:
        print(" Fair: Predictions have moderate error (RMSE < $200)")
    else:
        print("Poor: High prediction error (RMSE > $200)")
    
    # Interpret R²
    if metrics['R2'] > 0.95:
        print("Excellent model fit (R² > 0.95)")
    elif metrics['R2'] > 0.85:
        print(" Good model fit (R² > 0.85)")
    elif metrics['R2'] > 0.70:
        print("  Fair model fit (R² > 0.70)")
    else:
        print(" Poor model fit (R² < 0.70)")
    
    # Interpret Direction Accuracy
    if metrics['Direction_Accuracy'] > 60:
        print(f"Good trend prediction ({metrics['Direction_Accuracy']:.1f}% direction accuracy)")
    else:
        print(f" Weak trend prediction ({metrics['Direction_Accuracy']:.1f}% direction accuracy)")

# Run the evaluation pipeline if evaluate.py is executed directly
if __name__ == "__main__":
    from data_preprocessor import preprocess_pipeline
    from tensorflow import keras
    import config

    print("\n=== RUNNING MODEL EVALUATION ===\n")

    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()

    model = keras.models.load_model(config.MODEL_PATH)

    y_actual, y_pred = make_predictions(model, X_test, y_test, scaler)

    metrics = calculate_metrics(y_actual, y_pred)

    print_metrics(metrics)
