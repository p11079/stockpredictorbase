import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_and_prepare(df):
    """Cleans raw stock DataFrame and adds features/target."""
    # 1. Index and sorting
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 2. Missing values
    df = df.ffill().interpolate()

    # 3. Returns
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 4. Technical indicators
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 6. Lag features
    for lag in [1, 2, 3]:
        df[f'Close_lag{lag}'] = df['Close'].shift(lag)

    # 7. Target (next day close)
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with NaNs (handles the last row Target=NaN, and initial rolling/lag NaNs)
    df = df.dropna()

    return df


def scale_features(df, feature_columns):
    """Scales specified features and returns scaled array and scaler object."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_columns])
    return scaled_array, scaler


if __name__ == "__main__":
    # Generate dummy stock data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)

    # Cumulative sum to simulate realistic price walks
    close_prices = 100 + np.random.randn(100).cumsum()

    raw_df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100),
        'High': close_prices + np.abs(np.random.randn(100)),
        'Low': close_prices - np.abs(np.random.randn(100)),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, size=100)
    }, index=dates)

    print("Cleaning data...")
    cleaned_df = clean_and_prepare(raw_df)

    print("\nCleaned DataFrame (first 3 rows):")
    print(cleaned_df[['Close', 'MA_10', 'RSI', 'Target']].head(3))

    print(f"\nFinal shape after dropping NaNs: {cleaned_df.shape}")

    # Demonstrate scaling
    features_to_scale = ['Close', 'Volume', 'MA_10', 'RSI']
    scaled_data, fitted_scaler = scale_features(cleaned_df, features_to_scale)
    print(f"\nScaled Data Shape: {scaled_data.shape}")