"""Forecasting helpers."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_model(model_path: str):
    """Load a saved model from disk."""
    return joblib.load(model_path)


def predict(model, X) -> pd.Series:
    """Generate predictions with the supplied model."""
    return pd.Series(model.predict(X), index=X.index)


def save_forecast_plot(forecast_df: pd.DataFrame, output_path: str) -> str:
    """Create a simple time-series plot for the forecast and save it to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_df = forecast_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["date"], plot_df["forecast"], marker="o", color="#2c6cbf", linewidth=2)
    ax.set_title("Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted sales")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)
