import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date

# Local module imports
from stock_fetcher import fetch_stock_data
from data_cleaner import clean_and_prepare, scale_features
from stock_model import train_and_predict, predict_future
from evaluator import directional_accuracy
from utils import add_business_days

# 1. Page Configuration
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# 2. Title and Description
st.title("📈 Stock Price Prediction with Random Forest")
st.markdown("Predict future stock prices using historical OHLCV data and technical indicators.")

# 3. Sidebar Inputs
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date(2026, 1, 1))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

if st.sidebar.button("Run Prediction"):

    # 4. Main Pipeline - Fetch Data
    with st.spinner(f"Fetching data for {ticker}..."):
        df_raw = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if df_raw is None or df_raw.empty:
        st.error(f"No data found for ticker '{ticker}'. Please check the symbol and date range.")
        st.stop()

    if len(df_raw) < 100:
        st.warning(f"Data contains only {len(df_raw)} rows. The model requires more data for optimal performance.")

    # 5. Clean and Prepare
    with st.spinner("Processing data..."):
        df_clean = clean_and_prepare(df_raw.copy())

        desired_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns',
                            'Log_Returns', 'MA_10', 'MA_50', 'RSI',
                            'Close_lag1', 'Close_lag2', 'Close_lag3']

        # Only include columns successfully created
        actual_features = [f for f in desired_features if f in df_clean.columns]

        # 6. Scale Features
        scaled_array, scaler = scale_features(df_clean, actual_features)

        # Rebuild DataFrame to retain feature names for the model
        df_scaled = pd.DataFrame(scaled_array, columns=actual_features, index=df_clean.index)
        df_scaled['Target'] = df_clean['Target'].values

    # 7. Train Model
    with st.spinner("Training Random Forest model..."):
        preds, actuals, model = train_and_predict(df_scaled, actual_features, 'Target')

    # 8. Compute Metrics
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    safe_actuals = np.where(actuals == 0, 1e-8, actuals)  # Prevent division by zero
    mape = np.mean(np.abs((actuals - preds) / safe_actuals)) * 100
    dir_acc = directional_accuracy(actuals, preds)

    # 9. Display Metrics
    st.subheader("Model Evaluation (Test Set)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"{rmse:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("MAPE", f"{mape:.2f}%")
    col4.metric("Directional Accuracy", f"{dir_acc:.2f}%")

    # 10. Predict Next Business Day
    st.subheader("Forecast")
    latest_features_df = df_scaled[actual_features].iloc[-1:]  # Pass as DataFrame
    next_price_pred = predict_future(model, latest_features_df)

    last_date = df_clean.index[-1].to_pydatetime()
    next_date = add_business_days(last_date, 1)

    st.metric(label=f"Predicted Close for {next_date.strftime('%Y-%m-%d')}",
              value=f"${next_price_pred:.2f}")

    # 11. Visualizations
    st.subheader("Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Closing Price", "Train/Test Split", "Actual vs Predicted", "Residuals"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df_clean.index, df_clean['Close'], color='blue')
        ax1.set_title("Closing Price Over Time")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with tab2:
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(train_df.index, train_df['Close'], color='blue', label="Train")
        ax2.plot(test_df.index, test_df['Close'], color='orange', label="Test")
        ax2.set_title("Train-Test Split")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    with tab3:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(actuals, label="Actual", color='blue')
        ax3.plot(preds, label="Predicted", color='orange', linestyle='--')
        ax3.set_title("Actual vs Predicted (Test Set)")
        ax3.set_xlabel("Time Step (Test Set)")
        ax3.set_ylabel("Price")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    with tab4:
        residuals = actuals - preds
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.hist(residuals, bins=30, color='red', alpha=0.6, edgecolor='black')
        ax4.axvline(0, color='black', linestyle='--')
        ax4.set_title("Prediction Residuals Distribution")
        ax4.set_xlabel("Error (Actual - Predicted)")
        ax4.set_ylabel("Frequency")
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

    # 12. Optional Expander
    with st.expander("View Processed Data (First 50 Rows)"):
        st.dataframe(df_clean.head(50))