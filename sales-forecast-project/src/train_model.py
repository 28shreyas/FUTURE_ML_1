"""Model training entry points."""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_prep import clean_sales_dataframe, load_raw_data
from evaluate import build_error_summary, evaluate_model
from features import create_time_features


def build_training_pipeline() -> Pipeline:
    """Build a standard training pipeline with a random forest model."""
    return Pipeline(
        [
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )


def train_baseline_model(X, y):
    """Train a baseline model using a standardized pipeline."""
    model = build_training_pipeline()
    model.fit(X, y)
    return model


def save_model(model, output_path: str) -> None:
    """Persist a trained model to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def prepare_training_data(df: pd.DataFrame, date_column: str = "Month", target_column: str = "Sales"):
    """Prepare feature matrix and target vector from raw sales data."""
    cleaned_df = clean_sales_dataframe(df, date_column=date_column, target_column=target_column)
    engineered_df = create_time_features(cleaned_df, date_column)

    feature_candidates = [column for column in engineered_df.columns if column not in {date_column, target_column}]
    numeric_features = engineered_df[feature_candidates].select_dtypes(include=["number"])

    if numeric_features.empty:
        raise ValueError("No numeric feature columns found after preprocessing.")

    X = numeric_features
    y = engineered_df[target_column]
    return X, y


def find_default_input_path(root_path: Path) -> Path:
    raw_dir = root_path / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {raw_dir}. "
            "Create the folder and add a CSV file, or pass --input-path explicitly."
        )

    default_file = raw_dir / "sales.csv"
    if default_file.exists():
        return default_file

    csv_files = sorted(raw_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"No CSV files found in {raw_dir}. "
        "Create a raw dataset CSV or pass --input-path explicitly."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sales forecasting baseline model.")
    root_path = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to the raw sales CSV input file. If omitted, searches data/raw/ for a CSV.",
    )
    parser.add_argument(
        "--model-output",
        default=str(root_path / "models" / "baseline_model.joblib"),
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--date-column",
        default="Month",
        help="Name of the date column in the raw data.",
    )
    parser.add_argument(
        "--target-column",
        default="Sales",
        help="Name of the target column in the raw data.",
    )
    args = parser.parse_args()
    if args.input_path is None:
        args.input_path = str(find_default_input_path(root_path))
    return args


def train_and_evaluate(
    X, y, output_path: str, test_size: float = 0.2, random_state: int = 42
):
    """Train on a split dataset and return a model plus evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = train_baseline_model(X_train, y_train)
    save_model(model, output_path)

    train_preds = pd.Series(model.predict(X_train), index=y_train.index)
    test_preds = pd.Series(model.predict(X_test), index=y_test.index)

    return (
        model,
        evaluate_model(y_train, train_preds),
        evaluate_model(y_test, test_preds),
        build_error_summary(y_train, train_preds),
        build_error_summary(y_test, test_preds),
    )


def generate_forecast(df: pd.DataFrame, date_column: str = "Month", target_column: str = "Sales", horizon: int = 6):
    """Generate a short-horizon forecast from the latest sales history."""
    cleaned_df = clean_sales_dataframe(df, date_column=date_column, target_column=target_column)
    engineered_df = create_time_features(cleaned_df, date_column)
    X, y = prepare_training_data(engineered_df, date_column=date_column, target_column=target_column)
    model = train_baseline_model(X, y)

    last_row = engineered_df.iloc[-1]
    feature_columns = X.columns.tolist()
    future_rows = []
    last_date = pd.Timestamp(engineered_df[date_column].max())

    for offset in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=offset)
        row = {
            "year": future_date.year,
            "month": future_date.month,
            "day": future_date.day,
            "day_of_week": future_date.dayofweek,
            "week_of_year": future_date.isocalendar().week,
            "quarter": future_date.quarter,
            "trend": len(engineered_df) + offset,
            "month_sin": np.sin(2 * np.pi * future_date.month / 12),
            "month_cos": np.cos(2 * np.pi * future_date.month / 12),
            "is_month_end": int(future_date.day > 25),
            "is_weekend": int(future_date.dayofweek in {5, 6}),
        }
        for column in feature_columns:
            if column in row:
                continue
            row[column] = last_row[column] if column in last_row.index else 0
        future_rows.append(row)

    future_features = pd.DataFrame(future_rows, columns=feature_columns)
    predictions = model.predict(future_features)
    return pd.DataFrame({"date": pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D"), "forecast": np.round(predictions, 2)})


def main():
    args = parse_args()
    raw_df = load_raw_data(args.input_path)
    X, y = prepare_training_data(raw_df, args.date_column, args.target_column)
    _, train_metrics, test_metrics, _, _ = train_and_evaluate(
        X, y, args.model_output, test_size=0.2, random_state=42
    )

    print(f"Saved model to: {args.model_output}")
    print("Training metrics:")
    for metric_name, metric_value in train_metrics.items():
        print(f" - {metric_name}: {metric_value:.4f}")

    print("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f" - {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
