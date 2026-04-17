import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set seaborn style for better aesthetics
sns.set_theme(style="whitegrid")


def _handle_save_or_show(save_path):
    """Helper function to save or display the plot."""
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_closing_price(df, title="Closing Price Over Time", save_path=None):
    """Plots the closing price over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], color='blue', label='Close Price')
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    _handle_save_or_show(save_path)


def plot_train_test_split(train_df, test_df, title="Train-Test Split", save_path=None):
    """Overlays train and test closing prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_df.index, train_df['Close'], color='blue', label='Train')
    plt.plot(test_df.index, test_df['Close'], color='orange', label='Test')
    plt.title(title, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    _handle_save_or_show(save_path)


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Plots actual versus predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')
    plt.title(title, fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    _handle_save_or_show(save_path)


def plot_residuals(y_true, y_pred, title="Residuals", save_path=None):
    """Plots a histogram and KDE of prediction residuals."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='red', bins=30)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(title, fontsize=14)
    plt.xlabel('Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    _handle_save_or_show(save_path)


if __name__ == "__main__":
    np.random.seed(42)

    # Generate dummy datetime index and closing prices
    dates = pd.date_range("2023-01-01", periods=150)
    close_prices = 100 + np.random.randn(150).cumsum()
    df = pd.DataFrame({'Close': close_prices}, index=dates)

    # Create train/test split dummy data
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]

    # Create dummy actual vs predicted arrays
    actuals = test_data['Close'].values
    predictions = actuals + np.random.randn(len(actuals)) * 2

    print("Generating sample plots...")

    # 1. Closing Price Plot (saving as file to demonstrate)
    plot_closing_price(df, save_path="sample_closing_price.png")

    # 2. Train-Test Split Plot (displaying)
    print("Displaying Train-Test Split...")
    plot_train_test_split(train_data, test_data)

    # 3. Actual vs Predicted Plot (displaying)
    print("Displaying Actual vs Predicted...")
    plot_actual_vs_predicted(actuals, predictions)

    # 4. Residuals Plot (displaying)
    print("Displaying Residuals...")
    plot_residuals(actuals, predictions)