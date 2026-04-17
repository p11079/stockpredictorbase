import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_regression(y_true, y_pred):
    """Calculates and prints RMSE, MAE, and MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Handle division by zero for MAPE
    safe_y_true = np.where(y_true == 0, 1e-8, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return rmse, mae, mape


def directional_accuracy(y_true, y_pred):
    """Calculates the % of matching up/down directions."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Compare direction of change from the previous actual step
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(y_pred[1:] - y_true[:-1])

    # Check where directions match (ignoring zero changes for strictness)
    matches = (actual_direction == predicted_direction)
    accuracy = np.mean(matches) * 100

    return accuracy


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Plots actuals vs predictions and saves to a file."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    file_name = 'actual_vs_predicted.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully as '{file_name}'")


if __name__ == "__main__":
    np.random.seed(42)

    # Generate dummy evaluation data
    actuals = np.linspace(100, 150, 50) + np.random.randn(50) * 2

    # Predictions roughly follow actuals with noise
    predictions = actuals + np.random.randn(50) * 1.5

    print("--- Regression Metrics ---")
    evaluate_regression(actuals, predictions)

    print("\n--- Directional Accuracy ---")
    dir_acc = directional_accuracy(actuals, predictions)
    print(f"Directional Accuracy: {dir_acc:.2f}%")

    print("\n--- Generating Plot ---")
    plot_actual_vs_predicted(actuals, predictions)