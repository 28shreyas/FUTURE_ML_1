import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_prep import load_raw_data
from src.train_model import generate_forecast, prepare_training_data


class ForecastPipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.data_path = self.root / "data" / "raw" / "sales.csv"

    def test_load_raw_data_cleans_dates_and_missing_values(self):
        df = load_raw_data(str(self.data_path))

        self.assertIn("Month", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["Month"]))
        self.assertFalse(df.isna().any().any())

    def test_prepare_training_data_builds_engineered_features(self):
        df = load_raw_data(str(self.data_path))
        X, y = prepare_training_data(df, date_column="Month", target_column="Sales")

        self.assertEqual(len(X), len(y))
        self.assertFalse(X.isnull().any().any())
        self.assertTrue({"year", "month", "day_of_week", "trend"}.issubset(set(X.columns)))

    def test_generate_forecast_returns_future_periods(self):
        df = load_raw_data(str(self.data_path))
        forecast_df = generate_forecast(df, date_column="Month", target_column="Sales", horizon=6)

        self.assertEqual(len(forecast_df), 6)
        self.assertIn("forecast", forecast_df.columns)
        self.assertIn("date", forecast_df.columns)

    def test_load_raw_data_accepts_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.json"
            payload = [
                {"date": "2021-01-01", "sales": 120.0, "customers": 10},
                {"date": "2021-01-02", "sales": 130.0, "customers": 11},
            ]
            path.write_text(json.dumps(payload), encoding="utf-8")

            df = load_raw_data(str(path))

            self.assertIn("date", df.columns)
            self.assertIn("sales", df.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))

    def test_load_raw_data_accepts_excel_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            payload = pd.DataFrame(
                [{"date": "2021-01-01", "sales": 120.0, "customers": 10}]
            )
            payload.to_excel(path, index=False)

            df = load_raw_data(str(path))

            self.assertIn("date", df.columns)
            self.assertIn("sales", df.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))


if __name__ == "__main__":
    unittest.main()
