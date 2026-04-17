import pickle
from datetime import datetime, timedelta


def validate_date(date_string):
    """Validates YYYY-MM-DD format, returns datetime object."""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Incorrect date format for '{date_string}', should be YYYY-MM-DD")


def save_model(model, filepath="model.pkl"):
    """Saves model to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath="model.pkl"):
    """Loads model from disk."""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def get_today_date():
    """Returns today's date as YYYY-MM-DD string."""
    return datetime.today().strftime("%Y-%m-%d")


def add_business_days(start_date, days):
    """Adds business days (skips weekends) to a datetime object."""
    current_date = start_date
    added = 0
    while added < days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # 0 to 4 is Monday to Friday
            added += 1
    return current_date


if __name__ == "__main__":
    print("--- Testing validate_date ---")
    valid_date = validate_date("2023-01-15")
    print(f"Valid date parsed: {valid_date.date()}")

    try:
        validate_date("15-01-2023")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Testing get_today_date ---")
    today = get_today_date()
    print(f"Today is: {today}")

    print("\n--- Testing add_business_days ---")
    # Friday, Jan 13, 2023
    start = datetime(2023, 1, 13)
    next_biz_day = add_business_days(start, 1)
    # Should skip Sat/Sun and return Monday, Jan 16, 2023
    print(f"1 business day after {start.date()} is {next_biz_day.date()}")

    print("\n--- Testing save_model and load_model ---")
    dummy_model = {"weights": [0.5, -0.2, 0.1], "name": "DummyRegressor"}
    save_model(dummy_model, "test_model.pkl")

    loaded = load_model("test_model.pkl")
    print(f"Loaded model data: {loaded}")