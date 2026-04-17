import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_and_predict(df, feature_columns, target_column):
    """Splits data chronologically, trains a Random Forest, and predicts."""
    split_idx = int(len(df) * 0.8)

    X_train = df[feature_columns].iloc[:split_idx]
    y_train = df[target_column].iloc[:split_idx]
    X_test = df[feature_columns].iloc[split_idx:]
    y_test = df[target_column].iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions, y_test.values, model


def predict_future(model, latest_features):
    """Predicts next day value using latest features."""
    return model.predict(latest_features)[0]


if __name__ == "__main__":
    np.random.seed(42)
    n_rows = 150

    # Generate dummy data
    dummy_data = {
        'Close_lag1': np.random.randn(n_rows).cumsum(),
        'MA_10': np.random.randn(n_rows).cumsum(),
        'RSI': np.random.uniform(30, 70, n_rows),
    }
    df = pd.DataFrame(dummy_data)

    # Target is loosely based on lag1 + some noise
    df['Target'] = df['Close_lag1'] + np.random.randn(n_rows) * 0.5

    features = ['Close_lag1', 'MA_10', 'RSI']

    print("Training model...")
    preds, actuals, trained_model = train_and_predict(df, features, 'Target')

    print(f"Test Set Size: {len(actuals)}")
    print(f"Sample Predictions: {preds[:3]}")
    print(f"Sample Actuals:     {actuals[:3]}")

    # FIXED: Passed as DataFrame with matching feature names
    latest_feat = pd.DataFrame([[
        df['Close_lag1'].iloc[-1],
        df['MA_10'].iloc[-1],
        df['RSI'].iloc[-1]
    ]], columns=features)

    future_pred = predict_future(trained_model, latest_feat)

    print(f"\nNext Day Prediction: {future_pred:.4f}")