"""Streamlit dashboard for StockPredictorBase."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from data_cleaner import clean_and_prepare, scale_features
from evaluator import directional_accuracy, evaluate_regression, plot_actual_vs_predicted, plot_residuals
from stock_fetcher import fetch_stock_data
from stock_model import predict_future, train_and_predict
from stock_vizualizer import plot_closing_price, plot_train_test_split
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

    plot_closing_price(df, title=f"{ticker} closing price")
    st.pyplot(st.pyplot())

    plot_train_test_split(train_df, test_df, title=f"{ticker} train/test split")
    st.pyplot(st.pyplot())

    plot_actual_vs_predicted(actuals, predictions, title=f"{ticker} actual vs predicted")
    st.pyplot(st.pyplot())

    plot_residuals(actuals, predictions, title=f"{ticker} residuals")
    st.pyplot(st.pyplot())
else:
    st.info("Choose a ticker and date range, then run the forecast.")
