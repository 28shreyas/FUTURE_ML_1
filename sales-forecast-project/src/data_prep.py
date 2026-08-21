"""Data loading and preprocessing utilities."""

import json
import re
from pathlib import Path
import pandas as pd

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


def _detect_date_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Choose the most likely date column from a dataframe."""
    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        col for col in df.columns if any(token in str(col).lower() for token in ["date", "time", "month", "period", "day", "ds", "timestamp"])
    ]
    if candidates:
        return candidates[0]
    return df.columns[0]


def _detect_target_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    """Choose the most likely target column from a dataframe."""
    if preferred and preferred in df.columns:
        return preferred

    normalized_columns = {str(col).lower(): col for col in df.columns}
    for token in ["sales", "revenue", "target", "amount", "y", "forecast", "value"]:
        if token in normalized_columns:
            return normalized_columns[token]

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            return column

    return df.columns[-1]


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract readable text from a PDF, if the optional dependency is available."""
    if fitz is None:
        return ""

    try:
        doc = fitz.open(file_path)
        text_chunks = [page.get_text() for page in doc]
        return "\n".join(chunk for chunk in text_chunks if chunk)
    except Exception:
        return ""


def _extract_text_from_image(file_path: Path) -> str:
    """Extract text from an image using OCR if available."""
    if Image is None or pytesseract is None:
        return ""

    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def _text_to_dataframe(text: str) -> pd.DataFrame:
    """Attempt to infer a dataframe from extracted text by splitting on whitespace and commas."""
    if not text:
        raise ValueError("No extractable text found.")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No extractable text found.")

    rows = []
    for line in lines:
        if "," in line:
            parts = [item.strip() for item in line.split(",")]
        else:
            parts = re.split(r"\s{2,}|\t", line)
        if parts:
            rows.append(parts)

    if not rows:
        raise ValueError("No tabular structure found in extracted text.")

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    if not data_rows:
        data_rows = [header]
    df = pd.DataFrame(data_rows, columns=header)
    return df


def _read_tabular_file(file_path: Path) -> pd.DataFrame:
    """Load a dataframe from CSV, Excel, JSON, PDF, image, or other common tabular formats."""
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(file_path, sep=None, engine="python")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    if suffix == ".json":
        try:
            return pd.read_json(file_path)
        except ValueError:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
            raise

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)

    if suffix == ".pdf":
        text = _extract_text_from_pdf(file_path)
        if text:
            return _text_to_dataframe(text)
        raise ValueError("Could not extract readable text from the uploaded PDF.")

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
        text = _extract_text_from_image(file_path)
        if text:
            return _text_to_dataframe(text)
        raise ValueError("Could not extract readable text from the uploaded image.")

    return pd.read_csv(file_path, sep=None, engine="python")


def clean_sales_dataframe(
    df: pd.DataFrame, date_column: str = "Month", target_column: str = "Sales"
) -> pd.DataFrame:
    """Clean and standardize a sales table for forecasting."""
    cleaned = df.copy()

    if date_column not in cleaned.columns:
        raise ValueError(f"Date column '{date_column}' not found in input data.")
    if target_column not in cleaned.columns:
        raise ValueError(f"Target column '{target_column}' not found in input data.")

    cleaned[date_column] = pd.to_datetime(cleaned[date_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)

    for column in cleaned.columns:
        if column == date_column:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in cleaned.columns:
        if column == date_column:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    cleaned = cleaned.drop_duplicates(subset=[date_column], keep="last")
    cleaned = cleaned.sort_values(date_column).reset_index(drop=True)
    return cleaned


def load_raw_data(path: str, date_column: str | None = None, target_column: str | None = None) -> pd.DataFrame:
    """Load sales data from common file formats and return a cleaned dataframe."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}. "
            "Create the file or pass a valid --input-path."
        )

    df = _read_tabular_file(file_path)
    resolved_date_column = _detect_date_column(df, date_column)
    resolved_target_column = _detect_target_column(df, target_column)

    return clean_sales_dataframe(df, date_column=resolved_date_column, target_column=resolved_target_column)


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned data to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
