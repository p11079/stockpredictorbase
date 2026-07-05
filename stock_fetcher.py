import yfinance as yf
from difflib import SequenceMatcher

EXCHANGE_SUFFIXES = {
    "US": "",
    "NSE": ".NS",
    "BSE": ".BO",
    "JSE": ".JO",
    "LSE": ".L",
    "ASX": ".AX",
    "HKEX": ".HK",
    "TSE": ".T",
    "SGX": ".SI",
    "FRA": ".DE",
    "EURONEXT": ".PA",
}

EXCHANGE_SYMBOL_HINTS = {
    "US": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"],
    "NSE": ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "HDFCBANK", "ICICIBANK", "LT", "WIPRO", "AXISBANK"],
    "BSE": ["RELIANCE", "TCS", "INFY", "SBIN", "ITC", "HDFCBANK", "ICICIBANK", "LT", "WIPRO", "AXISBANK"],
    "JSE": ["NPN", "SOL", "SBK", "FSR", "BID", "ANG", "MTN", "CLS", "REM", "GFI"],
    "LSE": ["VOD", "HSBA", "BP", "RIO", "AZN", "ULVR", "GSK", "BARC", "SHEL", "BATS"],
    "ASX": ["BHP", "CBA", "CSL", "WOW", "WBC", "NAB", "ANZ", "MQG", "RIO", "TLS"],
    "HKEX": ["0700", "9988", "3690", "0939", "1299", "2269", "0005", "0388", "1211", "2318"],
    "TSE": ["7203", "6758", "9984", "8306", "9432", "9433", "7974", "6861", "6098", "8035"],
    "SGX": ["D05", "C6L", "U11", "O39", "G20", "Z74", "S68", "C09", "V03", "A17U"],
    "FRA": ["SIE", "AIR", "SAP", "ALV", "BMW", "DTE", "RWE", "BAS", "MBG", "VOW3"],
    "EURONEXT": ["AIR", "MC", "BNP", "SAN", "OR", "AI", "DG", "SU", "GLE", "KER"],
}


def _score_query(query, candidate):
    query = query.upper().strip()
    candidate = candidate.upper().strip()
    if candidate == query:
        return 1.0
    if candidate.startswith(query):
        return 0.95
    if query in candidate:
        return 0.9
    return SequenceMatcher(None, query, candidate).ratio()


def suggest_tickers(query, exchange="US", limit=5):
    """Return closest exchange-specific ticker suggestions."""
    exchange = str(exchange).upper().strip() if exchange else "US"
    candidates = EXCHANGE_SYMBOL_HINTS.get(exchange, [])
    query = str(query).upper().strip()
    if not query:
        return []

    scored = sorted(
        candidates,
        key=lambda candidate: (-_score_query(query, candidate), len(candidate), candidate),
    )
    return [normalize_ticker(symbol, exchange=exchange) for symbol in scored[:limit]]


def normalize_ticker(ticker, exchange="US"):
    """Normalize a ticker for Yahoo Finance based on an exchange code."""
    ticker = str(ticker).upper().strip()
    exchange = str(exchange).upper().strip() if exchange else "US"
    suffix = EXCHANGE_SUFFIXES.get(exchange)
    if suffix is None:
        raise ValueError(f"Unsupported exchange '{exchange}'.")
    if not ticker:
        raise ValueError("Ticker symbol is required.")
    if suffix and not ticker.endswith(suffix):
        ticker = f"{ticker}{suffix}"
    return ticker

def fetch_stock_data(ticker, start_date, end_date, exchange="US"):
    """Fetches OHLCV data for a given ticker."""
    try:
        ticker = normalize_ticker(ticker, exchange=exchange)
        if not ticker:
            print("Error: ticker symbol is required.")
            return None

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            print(f"Error: No data found or invalid ticker '{ticker}'.")
            return None

        # yfinance can return either flat columns or a multi-index with ticker
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [column for column in required_columns if column not in data.columns]
        if missing:
            print(f"Error: missing expected columns {missing}.")
            return None

        return data[required_columns].copy()

    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None


if __name__ == "__main__":
    sample_ticker = "AAPL"
    print(f"Fetching data for {sample_ticker}...")

    df = fetch_stock_data(sample_ticker, "2023-01-01", "2023-01-10")

    if df is not None:
        print(df.head())
