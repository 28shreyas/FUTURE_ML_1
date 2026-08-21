import tempfile
import unittest
from pathlib import Path

from src.data_prep import load_raw_data


class PdfImageIngestionTests(unittest.TestCase):
    def test_pdf_text_fallback_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.pdf"
            path.write_bytes(b"not a real pdf")

            with self.assertRaises(ValueError):
                load_raw_data(str(path))

    def test_image_text_fallback_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.png"
            path.write_bytes(b"not a real image")

            with self.assertRaises(ValueError):
                load_raw_data(str(path))


if __name__ == "__main__":
    unittest.main()
