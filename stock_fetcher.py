import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    """Fetches OHLCV data for a given ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            print(f"Error: No data found or invalid ticker '{ticker}'.")
            return None

        return data[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None


if __name__ == "__main__":
    sample_ticker = "AAPL"
    print(f"Fetching data for {sample_ticker}...")

    df = fetch_stock_data(sample_ticker, "2023-01-01", "2023-01-10")

    if df is not None:
        print(df.head())