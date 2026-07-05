"""Streamlit dashboard for StockPredictorBase."""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from data_cleaner import clean_and_prepare, scale_features
from evaluator import directional_accuracy, evaluate_regression
from stock_fetcher import fetch_stock_data
from stock_model import predict_future, train_and_predict
from utils import add_business_days

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

sns.set_theme(style="whitegrid")
st.set_page_config(page_title="StockPredictorBase", page_icon="\U0001F4C8", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .muted { color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>StockPredictorBase</h1>
      <p class="muted">Explore a stock forecast built from historical data, technical features, and a Random Forest model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _plot_price_history(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], color="#2563eb", linewidth=2)
    ax.set_title(f"{ticker} closing price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    return fig


def _plot_split(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train_df.index, train_df["Close"], color="#2563eb", label="Train", linewidth=2)
    ax.plot(test_df.index, test_df["Close"], color="#f97316", label="Test", linewidth=2)
    ax.set_title(f"{ticker} train/test split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_actual_vs_predicted(actuals, predictions, ticker: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actuals, label="Actual", color="#2563eb", linewidth=2)
    ax.plot(predictions, label="Predicted", color="#f97316", linestyle="--", linewidth=2)
    ax.set_title(f"{ticker} actual vs predicted")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_residuals(actuals, predictions, ticker: str):
    residuals = pd.Series(actuals) - pd.Series(predictions)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True, color="#dc2626", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} residuals")
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
with col2:
    start_date = st.date_input("Start date", value=date(2023, 1, 1))
with col3:
    end_date = st.date_input("End date", value=date.today())

run_button = st.button("Run forecast", type="primary")

if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner(f"Fetching {ticker} data..."):
        raw_df = fetch_stock_data(ticker, start_date.isoformat(), end_date.isoformat())

    if raw_df is None or raw_df.empty:
        st.error("No market data was returned for that ticker and date range.")
        st.stop()

    if len(raw_df) < 100:
        st.warning("This range has fewer than 100 rows. Model quality may be weaker than usual.")

    df = clean_and_prepare(raw_df.copy())
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    feature_columns = [column for column in FEATURE_COLUMNS if column in df.columns]

    if missing:
        st.info(f"Missing feature columns were skipped: {', '.join(missing)}")

    if len(df) < 40 or len(feature_columns) < 5:
        st.error("Not enough prepared data to train and evaluate the model.")
        st.stop()

    scaled_array, _ = scale_features(df, feature_columns)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns, index=df.index)
    scaled_df[TARGET_COLUMN] = df[TARGET_COLUMN].values

    predictions, actuals, model = train_and_predict(scaled_df, feature_columns, TARGET_COLUMN)
    rmse, mae, mape = evaluate_regression(actuals, predictions)
    dir_acc = directional_accuracy(actuals, predictions)

    latest_features = scaled_df[feature_columns].iloc[-1:].values
    next_day_pred = predict_future(model, latest_features)
    next_date = add_business_days(df.index[-1], 1)

    st.subheader(f"{ticker} forecast summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted next close", f"${next_day_pred:,.2f}")
    metric_cols[1].metric("RMSE", f"{rmse:,.2f}")
    metric_cols[2].metric("MAE", f"{mae:,.2f}")
    metric_cols[3].metric("Directional accuracy", f"{dir_acc:,.1f}%")

    st.caption(f"Next business day forecast for {next_date.date().isoformat()}")

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    st.pyplot(_plot_price_history(df, ticker), clear_figure=True)
    st.pyplot(_plot_split(train_df, test_df, ticker), clear_figure=True)
    st.pyplot(_plot_actual_vs_predicted(actuals, predictions, ticker), clear_figure=True)
    st.pyplot(_plot_residuals(actuals, predictions, ticker), clear_figure=True)
else:
    st.info("Choose a ticker and date range, then run the forecast.")
