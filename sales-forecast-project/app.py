from pathlib import Path
import sys

from flask import Flask, request, render_template_string, send_from_directory
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train_model import generate_forecast, prepare_training_data, train_and_evaluate
from data_prep import load_raw_data
from forecast import save_forecast_plot
from generate_sales_data import generate_sales_csv

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = ROOT / "data" / "raw"
app.config["MODEL_OUTPUT"] = ROOT / "models" / "web_baseline_model.joblib"
app.config["FIGURES_DIR"] = ROOT / "reports" / "figures"
app.config["FIGURES_DIR"].mkdir(parents=True, exist_ok=True)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sales Forecast </title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #15304b;
      --muted: #607086;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --border: #dce7f3;
    }
    * {box-sizing: border-box;}
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      margin: 0;
      background: linear-gradient(135deg, #f8fbff 0%, var(--bg) 100%);
      color: var(--text);
    }
    .page {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    .hero {
      background: linear-gradient(120deg, #112d4e 0%, #2563eb 100%);
      color: white;
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 12px 30px rgba(21, 48, 75, 0.16);
      margin-bottom: 20px;
    }
    .hero h1 {margin: 0 0 8px; font-size: 30px;}
    .hero p {margin: 0; opacity: 0.95; max-width: 760px;}
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 8px 24px rgba(21, 48, 75, 0.06);
      margin-bottom: 18px;
    }
    form {margin: 0;}
    label {display:block; margin: 10px 0 6px; font-weight: 600; color: var(--text);}
    input[type="text"], input[type="file"] {
      width: 100%;
      max-width: 420px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fbfdff;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border: none;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    button:hover {transform: translateY(-1px); box-shadow: 0 6px 14px rgba(37, 99, 235, 0.2);}
    .secondary-btn {background: #0f766e;}
    .grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;}
    .card {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 16px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    }
    .card h3 {margin-top: 0; margin-bottom: 8px; font-size: 17px;}
    .card ul {margin: 0; padding-left: 18px; color: var(--muted);}
    .forecast-table {border-collapse: collapse; width: 100%; margin-top: 12px;}
    .forecast-table th, .forecast-table td {border: 1px solid var(--border); padding: 10px; text-align: left;}
    .forecast-table th {background: var(--accent-soft); color: var(--accent);}
    img {max-width: 100%; border: 1px solid var(--border); border-radius: 12px; margin-top: 16px; display:block;}
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 12px;
      border-radius: 10px;
      overflow-x: auto;
      font-size: 13px;
    }
    .pill {display:inline-block; margin-top: 10px; padding: 6px 10px; border-radius: 999px; background: rgba(255,255,255,0.16); font-size: 13px;}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Sales Forecast Studio</h1>
      <p>Upload sales data in CSV, Excel, JSON, PDF, or image formats and generate a polished forecast with evaluation insights in seconds.</p>
      <span class="pill">Clean • Engineer • Forecast • Visualize</span>
    </div>

    <div class="panel">
      <form method="post" enctype="multipart/form-data">
        <label>Upload data file
          <input type="file" name="dataset_file" accept=".csv,.json,.xlsx,.xls,.parquet,.pq,.tsv,.txt,.pdf,.png,.jpg,.jpeg,.bmp,.gif,.tiff,.webp" />
        </label>
        <label>
          Date column name
          <input type="text" name="date_column" value="Month" />
        </label>
        <label>
          Target column name
          <input type="text" name="target_column" value="Sales" />
        </label>
        <button type="submit">Train model</button>
      </form>

      <form method="post" style="margin-top:16px;">
        <input type="hidden" name="generate_sample" value="1" />
        <button type="submit" class="secondary-btn">Generate sample dataset</button>
      </form>
    </div>

    {% if message %}
      <div class="panel">
        <h2 style="margin-top:0;">Result</h2>
        <p>{{ message }}</p>
      </div>
    {% endif %}

    {% if train_metrics or test_metrics %}
      <div class="grid">
        <div class="card">
          <h3>Training metrics</h3>
          <ul>
            {% for name, value in train_metrics.items() %}
              <li><strong>{{ name }}</strong>: {{ "%.4f" | format(value) }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="card">
          <h3>Test metrics</h3>
          <ul>
            {% for name, value in test_metrics.items() %}
              <li><strong>{{ name }}</strong>: {{ "%.4f" | format(value) }}</li>
            {% endfor %}
          </ul>
        </div>
        <div class="card">
          <h3>Error review</h3>
          <ul>
            <li><strong>Mean error</strong>: {{ "%.4f" | format(test_error_summary.mean_error) }}</li>
            <li><strong>Max error</strong>: {{ "%.4f" | format(test_error_summary.max_error) }}</li>
          </ul>
        </div>
      </div>
    {% endif %}

    {% if forecast_rows %}
      <div class="panel">
        <h3 style="margin-top:0;">Forecast preview</h3>
        <table class="forecast-table">
          <thead>
            <tr><th>Date</th><th>Forecast</th></tr>
          </thead>
          <tbody>
            {% for row in forecast_rows %}
              <tr><td>{{ row.date }}</td><td>{{ "%.2f" | format(row.forecast) }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
        {% if forecast_image_url %}
          <img src="{{ forecast_image_url }}" alt="Forecast chart" />
        {% endif %}
      </div>
    {% endif %}

    {% if coefficients %}
      <div class="panel">
        <h3 style="margin-top:0;">Feature importances</h3>
        <pre>{{ coefficients }}</pre>
      </div>
    {% endif %}

    {% if test_error_summary and test_error_summary.top_errors %}
      <div class="panel">
        <h3 style="margin-top:0;">Largest forecast errors</h3>
        <ul>
          {% for item in test_error_summary.top_errors %}
            <li>Row {{ item.index }}: actual {{ "%.2f" | format(item.actual) }}, predicted {{ "%.2f" | format(item.predicted) }}, error {{ "%.2f" | format(item.error) }}</li>
          {% endfor %}
        </ul>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


def run_training(df: pd.DataFrame, date_column: str, target_column: str, horizon: int = 6):
    X, y = prepare_training_data(df, date_column, target_column)
    model, train_metrics, test_metrics, train_error_summary, test_error_summary = train_and_evaluate(
        X, y, str(app.config["MODEL_OUTPUT"]), test_size=0.2, random_state=42
    )
    importances = model.named_steps["model"].feature_importances_
    importance_pairs = sorted(zip(X.columns, importances), key=lambda item: item[1], reverse=True)
    feature_importances = "\n".join(
        f"{name}: {value:.4f}" for name, value in importance_pairs
    )

    forecast_df = generate_forecast(df, date_column, target_column, horizon=horizon)
    forecast_image_path = app.config["FIGURES_DIR"] / "forecast.png"
    save_forecast_plot(forecast_df, str(forecast_image_path))

    forecast_rows = [
        {"date": row["date"].strftime("%Y-%m-%d"), "forecast": float(row["forecast"])}
        for _, row in forecast_df.iterrows()
    ]
    return (
        train_metrics,
        test_metrics,
        feature_importances,
        forecast_rows,
        "/reports/figures/forecast.png",
        train_error_summary,
        test_error_summary,
    )


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename: str):
    return send_from_directory(app.config["FIGURES_DIR"], filename)


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    train_metrics = None
    test_metrics = None
    coefficients = None
    forecast_rows = None
    forecast_image_url = None
    test_error_summary = None

    if request.method == "POST":
        if request.form.get("generate_sample"):
            generate_sales_csv(ROOT / "data" / "raw" / "sales.csv", rows=1000)
            message = "Generated sample dataset at data/raw/sales.csv."
        else:
            dataset_file = request.files.get("dataset_file")
            date_column = request.form.get("date_column", "Month")
            target_column = request.form.get("target_column", "Sales")

            if dataset_file and dataset_file.filename:
                temp_path = app.config["UPLOAD_FOLDER"] / dataset_file.filename
                dataset_file.save(temp_path)
                try:
                    raw_df = load_raw_data(str(temp_path), date_column=date_column, target_column=target_column)
                    message = f"Loaded uploaded file: {dataset_file.filename}."
                except Exception as exc:
                    message = f"The uploaded file could not be parsed into forecastable data: {exc}"
                    raw_df = load_raw_data(str(ROOT / "data" / "raw" / "sales.csv"))
            else:
                default_path = ROOT / "data" / "raw" / "sales.csv"
                raw_df = load_raw_data(str(default_path))
                message = f"Loaded default dataset from {default_path}."

            (
                train_metrics,
                test_metrics,
                coefficients,
                forecast_rows,
                forecast_image_url,
                _,
                test_error_summary,
            ) = run_training(raw_df, date_column, target_column)
            message += " Model trained and saved."

    return render_template_string(
        HTML_TEMPLATE,
        message=message,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        coefficients=coefficients,
        forecast_rows=forecast_rows,
        forecast_image_url=forecast_image_url,
        test_error_summary=test_error_summary,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
