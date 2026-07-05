"""
Stock Prediction Project – Main Entry Point
Run this script: python main.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from project modules
from stock_fetcher import fetch_stock_data, normalize_ticker
from data_cleaner import clean_and_prepare, scale_features
from stock_model import train_and_predict, predict_future
from evaluator import evaluate_regression, directional_accuracy
from stock_vizualizer import (
    plot_closing_price,
    plot_train_test_split,
    plot_actual_vs_predicted,
    plot_residuals,
)
from utils import validate_date, get_today_date, add_business_days

OUTPUT_DIR = Path("outputs")

# =============================================================================
# 1. User Input
# =============================================================================
def get_user_input():
    """Prompt user for ticker and date range."""
    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA): ").upper().strip()
    exchange = input("Enter exchange code (US, NSE, BSE, JSE, LSE, ASX, HKEX, TSE, SGX, FRA, EURONEXT): ").upper().strip() or "US"
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    return ticker, exchange, start_date, end_date


# =============================================================================
# 2. Main Pipeline
# =============================================================================
def main():
    print("=" * 60)
    print("STOCK PRICE PREDICTION – MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # --- Get and validate inputs ---
    ticker, exchange, start_str, end_str = get_user_input()
    try:
        start_dt = validate_date(start_str)
        end_dt = validate_date(end_str)
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date.")
        normalized_ticker = normalize_ticker(ticker, exchange=exchange)
    except ValueError as e:
        print(f"Date error: {e}")
        return

    # --- Fetch data ---
    print(f"\nFetching data for {normalized_ticker} from {start_str} to {end_str}...")
    raw_df = fetch_stock_data(ticker, start_str, end_str, exchange=exchange)
    if raw_df is None:
        print("Exiting due to data fetch error.")
        return

    print(f"Downloaded {len(raw_df)} rows of OHLCV data.")

    if len(raw_df) < 100:
        print("Warning: Less than 100 days of data. Model may be unreliable.")

    # --- Feature engineering & cleaning ---
    print("\nCleaning data and generating features...")
    df = clean_and_prepare(raw_df.copy())
    print(f"Data after feature engineering: {df.shape}")
    if df.empty:
        print("Not enough rows remain after feature engineering. Try a longer date range.")
        return

    # Define feature columns (must match those created in clean_and_prepare)
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'Log_Returns',
        'MA_10', 'MA_50', 'RSI',
        'Close_lag1', 'Close_lag2', 'Close_lag3'
    ]

    # Check that all feature columns exist (some may be missing if data too short)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        print(f"Warning: Missing feature columns {missing}. Removing them.")
        feature_columns = [col for col in feature_columns if col in df.columns]

    if len(feature_columns) < 3:
        print("Not enough usable feature columns were created. Try a longer date range.")
        return

    target_column = 'Target'

    # --- Scale features ---
    print("\nScaling features...")
    scaled_array, scaler = scale_features(df, feature_columns)

    # Build scaled DataFrame with same index and column names
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)
    scaled_df[target_column] = df[target_column].values  # Target is NOT scaled

    # --- Train/Test Split & Model Training ---
    print("\nTraining Random Forest model...")
    predictions, actuals, model = train_and_predict(scaled_df, feature_columns, target_column)

    # Determine split index to separate train/test periods for plotting
    split_idx = int(len(scaled_df) * 0.8)
    train_df = df.iloc[:split_idx]        # original (unscaled) for visualisation
    test_df = df.iloc[split_idx:]

    # --- Evaluation ---
    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE ON TEST SET")
    print("=" * 40)
    evaluate_regression(actuals, predictions)
    dir_acc = directional_accuracy(actuals, predictions)
    print(f"Directional Accuracy: {dir_acc:.2f}%")

    # --- Future Prediction (Next Business Day) ---
    latest_features = scaled_df[feature_columns].iloc[-1:]  # keep feature names for sklearn
    next_day_pred = predict_future(model, latest_features)

    # Calculate next business day from last date in data
    last_date = df.index[-1]
    next_biz_date = add_business_days(last_date, 1)
    print(f"\nPredicted closing price for {next_biz_date.date()} (next business day): ${next_day_pred:.2f}")

    # --- Visualisations ---
    print("\nGenerating plots... (close each figure to continue)")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 1. Closing price history
    plot_closing_price(df, title=f"{normalized_ticker} Closing Price", save_path=OUTPUT_DIR / "closing_price.png")

    # 2. Train/Test split overlay
    plot_train_test_split(train_df, test_df, title=f"{normalized_ticker} Train-Test Split", save_path=OUTPUT_DIR / "train_test_split.png")

    # 3. Actual vs Predicted (test set)
    plot_actual_vs_predicted(actuals, predictions, title=f"{normalized_ticker} Actual vs Predicted", save_path=OUTPUT_DIR / "actual_vs_predicted.png")

    # 4. Residual distribution
    plot_residuals(actuals, predictions, title=f"{normalized_ticker} Residuals", save_path=OUTPUT_DIR / "residuals.png")

    print(f"\nPipeline completed successfully. Plots saved in {OUTPUT_DIR.resolve()}.")


if __name__ == "__main__":
    main()
