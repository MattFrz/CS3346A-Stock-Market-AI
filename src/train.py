"""
train. py

Trains the LSTM model on preprocessed stock data and saves the trained model.
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config and preprocessing
from config import WINDOW_SIZE, EPOCHS, BATCH_SIZE, MODEL_PATH
from data_preprocessor import preprocess_pipeline


def build_model(window_size, n_features):
    """
    Builds and compiles the LSTM model. 
    
    Args:
        window_size: Number of time steps in input
        n_features: Number of features per time step
        
    Returns:
        Compiled Keras model
    """
    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    print(f"Input shape: ({window_size}, {n_features})")
    print(f"Window size: {window_size} days")
    print(f"Features: {n_features}")
    
    model = Sequential([
        # First LSTM layer
        LSTM(units=50, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(units=50),
        Dropout(0.2),
        
        # Output layer
        Dense(units=1)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the model with callbacks for early stopping and model checkpointing.
    
    Returns:
        Training history object
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    
    # Ensure models directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    return history


def save_training_history(history):
    """
    Saves training history to a text file.
    """
    history_path = MODEL_PATH.replace('.keras', '_history.txt')
    
    with open(history_path, 'w') as f:
        f.write("Training History\n")
        f.write("="*50 + "\n\n")
        
        for epoch in range(len(history.history['loss'])):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"  Loss: {history.history['loss'][epoch]:.6f}\n")
            f. write(f"  MAE: {history.history['mae'][epoch]:.6f}\n")
            f.write(f"  Val Loss: {history.history['val_loss'][epoch]:.6f}\n")
            f.write(f"  Val MAE: {history.history['val_mae'][epoch]:.6f}\n\n")
        
        f.write("\nBest Epoch:\n")
        best_epoch = np.argmin(history.history['val_loss']) + 1
        f.write(f"  Epoch: {best_epoch}\n")
        f.write(f"  Val Loss: {min(history.history['val_loss']):.6f}\n")
        f.write(f"  Val MAE: {history.history['val_mae'][best_epoch - 1]:.6f}\n")
    
    print(f"Training history saved to {history_path}")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*60)
    print(" " * 15 + "STOCK PRICE PREDICTION")
    print(" " * 20 + "MODEL TRAINING")
    print("="*60 + "\n")
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    result = preprocess_pipeline()
    
    if result is None:
        print("Error: Data preprocessing failed.")
        return
    
    X_train, X_test, y_train, y_test, scaler = result
    
    # Get number of features from data shape
    n_features = X_train.shape[2]
    print(f"\nDetected {n_features} features in the data")
    
    # Step 2: Build model
    print("\nStep 2: Building model...")
    model = build_model(WINDOW_SIZE, n_features)
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Step 4: Save results
    print("\nStep 4: Saving results...")
    print(f"Model saved to {MODEL_PATH}")
    
    save_training_history(history)
    
    # Print final metrics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Loss: {history.history['loss'][best_epoch - 1]:.6f}")
    print(f"Training MAE: {history.history['mae'][best_epoch - 1]:. 6f}")
    print(f"Validation Loss: {min(history.history['val_loss']):.6f}")
    print(f"Validation MAE: {history.history['val_mae'][best_epoch - 1]:.6f}")
    print("="*60 + "\n")
    
    print("✅ Training complete! Model ready for evaluation and prediction.")


if __name__ == "__main__":
    main()