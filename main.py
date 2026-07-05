"""
Stock Prediction Project - Main Entry Point
Run this script: python main.py
"""

from __future__ import annotations

import pandas as pd

from data_cleaner import clean_and_prepare, scale_features
from evaluator import directional_accuracy, evaluate_regression
from stock_fetcher import fetch_stock_data
from stock_model import predict_future, train_and_predict
from stock_vizualizer import (
    plot_actual_vs_predicted,
    plot_closing_price,
    plot_residuals,
    plot_train_test_split,
)
from utils import add_business_days, validate_date

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Returns",
    "Log_Returns",
    "MA_10",
    "MA_50",
    "RSI",
    "Close_lag1",
    "Close_lag2",
    "Close_lag3",
]
TARGET_COLUMN = "Target"


def get_user_input():
    """Prompt user for ticker and date range."""
    ticker = input("Enter stock ticker symbol (e.g., AAPL, TSLA): ").upper().strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    return ticker, start_date, end_date


def _baseline_metrics(actuals, train_last_value):
    baseline = [train_last_value] * len(actuals)
    rmse, mae, mape = evaluate_regression(actuals, baseline)
    return baseline, rmse, mae, mape


def main():
    print("=" * 60)
    print("STOCK PRICE PREDICTION - MACHINE LEARNING PIPELINE")
    print("=" * 60)

    ticker, start_str, end_str = get_user_input()
    if not ticker:
        print("Ticker symbol is required.")
        return

    try:
        start_dt = validate_date(start_str)
        end_dt = validate_date(end_str)
        if start_dt >= end_dt:
            raise ValueError("Start date must be before end date.")
    except ValueError as exc:
        print(f"Date error: {exc}")
        return

    print(f"\nFetching data for {ticker} from {start_str} to {end_str}...")
    raw_df = fetch_stock_data(ticker, start_str, end_str)
    if raw_df is None:
        print("Exiting due to data fetch error.")
        return

    print(f"Downloaded {len(raw_df)} rows of OHLCV data.")
    if len(raw_df) < 100:
        print("Warning: Less than 100 days of data. Model may be unreliable.")

    print("\nCleaning data and generating features...")
    df = clean_and_prepare(raw_df.copy())
    print(f"Data after feature engineering: {df.shape}")

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    feature_columns = [column for column in FEATURE_COLUMNS if column in df.columns]
    if missing:
        print(f"Warning: Missing feature columns {missing}. Removing them.")

    if len(df) < 40 or len(feature_columns) < 5:
        print("Not enough prepared data to train and evaluate the model.")
        return

    print("\nScaling features...")
    scaled_array, _ = scale_features(df, feature_columns)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)
    scaled_df[TARGET_COLUMN] = df[TARGET_COLUMN].values

    print("\nTraining Random Forest model...")
    predictions, actuals, model = train_and_predict(scaled_df, feature_columns, TARGET_COLUMN)

    split_idx = int(len(scaled_df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print("\n" + "=" * 40)
    print("MODEL PERFORMANCE ON TEST SET")
    print("=" * 40)
    rmse, mae, mape = evaluate_regression(actuals, predictions)
    dir_acc = directional_accuracy(actuals, predictions)
    print(f"Directional Accuracy: {dir_acc:.2f}%")

    baseline_predictions, baseline_rmse, baseline_mae, baseline_mape = _baseline_metrics(
        actuals,
        train_df["Close"].iloc[-1],
    )
    print("\nBaseline comparison using last observed close:")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Baseline MAE:  {baseline_mae:.4f}")
    print(f"Baseline MAPE: {baseline_mape:.2f}%")

    latest_features = scaled_df[feature_columns].iloc[-1:].values
    next_day_pred = predict_future(model, latest_features)
    last_date = df.index[-1]
    next_biz_date = add_business_days(last_date, 1)
    print(f"\nPredicted closing price for {next_biz_date.date()}: ${next_day_pred:.2f}")

    print("\nGenerating plots... (close each figure to continue)")
    plot_closing_price(df, title=f"{ticker} Closing Price")
    plot_train_test_split(train_df, test_df, title=f"{ticker} Train-Test Split")
    plot_actual_vs_predicted(actuals, predictions, title=f"{ticker} - Actual vs Predicted")
    plot_residuals(actuals, predictions, title=f"{ticker} - Residuals")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
