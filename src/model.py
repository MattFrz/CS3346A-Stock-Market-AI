"""
model.py

Defines the LSTM neural network used to predict the
next-day closing price of the S&P 500.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(sequence_length: int, num_features: int):
    """
    Builds and compiles the LSTM model.

    Parameters
    ----------
    sequence_length : int
        Number of time steps in each input sequence (e.g., 60 days).
    num_features : int
        Number of features per day (Close only = 1, or more if you
        added indicators like MA, RSI, MACD, etc.).

    Returns
    -------
    model : keras.Model
        A compiled Keras model ready for training.
    """

    model = Sequential()

    # 1st LSTM layer (short-term patterns)
    model.add(
        LSTM(
            units=50,
            return_sequences=True,              # we have more LSTM layers after this
            input_shape=(sequence_length, num_features)
        )
    )
    model.add(Dropout(0.2))

    # 2nd LSTM layer (medium-term patterns)
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # 3rd LSTM layer (long-term patterns)
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Dense layer to combine learned patterns
    model.add(Dense(units=25, activation="relu"))

    # Output layer: predicts a single next-day price (regression)
    model.add(Dense(units=1))

    # Compile model: regression → MSE loss, Adam optimizer
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]     # Mean Absolute Error for easier interpretation
    )

    return model
