# StockPredictorBase

StockPredictorBase predicts the next close price for a stock using historical OHLCV data, engineered technical indicators, and a Random Forest model.

## What Changed

- Added a Streamlit dashboard for interactive forecasting.
- Added exchange-aware ticker handling for major markets.
- Added ticker search suggestions based on the selected exchange.
- Hardened the CLI flow so short datasets and invalid tickers fail safely.
- Saved generated plots into `outputs/`.

## Where You Can Run It

You can run this project locally on Windows, macOS, or Linux as long as you have Python 3.10+ installed and internet access for Yahoo Finance data.

## Step-by-Step Guide

### 1. Open the project folder

Open a terminal in the repository root.

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### 5. Or run the CLI version

```bash
python main.py
```

### 6. Enter inputs

For the CLI, provide:

- ticker symbol, such as `AAPL`, `RELIANCE`, `0700`, or `7203`
- exchange code, such as `US`, `NSE`, `BSE`, `JSE`, `LSE`, `ASX`, `HKEX`, `TSE`, `SGX`, `FRA`, or `EURONEXT`
- start date in `YYYY-MM-DD`
- end date in `YYYY-MM-DD`

For the Streamlit app, pick:

- a region preset
- an exchange
- a ticker symbol
- a date range

## Example Tickers

- `US`: `AAPL`, `MSFT`, `TSLA`
- `NSE`: `RELIANCE`, `TCS`, `INFY`, `SBIN`
- `BSE`: `RELIANCE`, `HDFCBANK`, `TCS`
- `JSE`: `NPN`, `SBK`
- `LSE`: `VOD`, `HSBA`, `BP`
- `ASX`: `BHP`, `CBA`, `CSL`
- `HKEX`: `0700`, `9988`
- `TSE`: `7203`, `6758`
- `SGX`: `D05`, `C09`
- `FRA`: `SIE`, `AIR`
- `EURONEXT`: `AIR`, `MC`

## What the App Outputs

- test metrics
  - RMSE
  - MAE
  - MAPE
  - directional accuracy
- next-business-day predicted close
- plot images saved into `outputs/`

## Notes

- The model uses historical data only. It is not a live trading signal.
- Some exchange tickers require exact Yahoo Finance symbols.
- Very short date ranges can collapse after feature engineering, in which case the app will ask for a longer window.

## Project Structure

```bash
stockpredictorbase/
├── streamlit_app.py
├── main.py
├── stock_fetcher.py
├── data_cleaner.py
├── stock_model.py
├── evaluator.py
├── stock_vizualizer.py
├── utils.py
├── requirements.txt
└── CHANGELOG.md
```

## License

This project is for educational and development purposes.
