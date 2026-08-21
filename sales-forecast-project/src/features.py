"""Feature engineering helpers."""

import numpy as np
import pandas as pd


def create_time_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Create richer time-based features from a datetime column."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)

    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    df["week_of_year"] = df[date_column].dt.isocalendar().week.astype(int)
    df["quarter"] = df[date_column].dt.quarter
    df["trend"] = np.arange(1, len(df) + 1)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_month_end"] = (df["day"] > 25).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df
