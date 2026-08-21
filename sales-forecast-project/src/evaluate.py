"""Evaluation utilities for forecasting performance."""

import math
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    """Return commonly used regression metrics."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def build_error_summary(y_true, y_pred):
    """Summarize the largest residuals so users can inspect forecast errors."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    residuals = (y_pred - y_true).astype(float)

    top_errors = []
    for index in residuals.abs().sort_values(ascending=False).head(3).index:
        top_errors.append(
            {
                "index": int(index),
                "actual": round(float(y_true.loc[index]), 2),
                "predicted": round(float(y_pred.loc[index]), 2),
                "error": round(float(residuals.loc[index]), 2),
            }
        )

    return {
        "mean_error": round(float(residuals.mean()), 2),
        "max_error": round(float(residuals.abs().max()), 2),
        "top_errors": top_errors,
    }
