# StockPredictorBase

StockPredictorBase is a machine learning stock prediction project that analyzes historical market data, engineers technical features, and produces next-day forecasts.

## What It Does

- Fetches historical stock price data from Yahoo Finance
- Cleans and enriches the dataset with technical indicators
- Trains a Random Forest regression model
- Compares model output against a naive last-value baseline
- Visualizes training, testing, predictions, and residuals

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- yfinance
- Streamlit

## Project Structure

```bash
stockpredictorbase/
├── app.py              # Streamlit dashboard
├── main.py             # CLI pipeline
├── data_cleaner.py     # Feature engineering and scaling
├── evaluator.py        # Metrics and plots
├── stock_fetcher.py    # Yahoo Finance data loading
├── stock_model.py      # Model training and prediction
├── stock_vizualizer.py # Plot helpers
├── utils.py            # Validation and persistence helpers
└── requirements.txt    # Project dependencies
```

## Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Or run the CLI pipeline:

```bash
python main.py
```

## Notes

- The model uses historical data only. It is not a live trading signal.
- Forecast quality depends on the ticker, date range, and amount of available history.
- Short date ranges may produce weak or unstable results.
- The Streamlit dashboard shows the forecast, error metrics, and plots in a single screen.

## Features

- Historical stock analysis
- Technical feature engineering
- Model training and evaluation
- Baseline comparison
- Plotting for price, split, prediction, and residual views

## Future Improvements

- Add walk-forward backtesting
- Persist forecast history
- Add richer model diagnostics
- Support multiple models and feature sets

## License

This project is for educational and development purposes.
