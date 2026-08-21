from pathlib import Path
import csv
import random
import datetime
import math


def generate_sales_csv(path: Path, rows: int = 1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.date(2021, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Month", "Advertising", "Price", "Discount", "Customers", "Sales"])
        for i in range(rows):
            date = start + datetime.timedelta(days=i)
            advertising = round(200 + 10 * math.sin(2 * math.pi * (i / 30)) + random.gauss(0, 5), 2)
            price = round(20 + 2 * math.sin(2 * math.pi * (i / 365)) + random.gauss(0, 0.5), 2)
            discount = round(max(0, 5 + 2 * math.sin(2 * math.pi * (i / 14)) + random.gauss(0, 1)), 2)
            customers = round(300 + 20 * math.cos(2 * math.pi * (i / 30)) + random.gauss(0, 10))
            sales = round(0.5 * advertising + 2.5 * customers - 5 * price + 3 * discount + random.gauss(0, 20), 2)
            writer.writerow([date.isoformat(), advertising, price, discount, customers, sales])


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales.csv"
    generate_sales_csv(output_path, rows=1000)
    print(f"Generated {output_path} with 1000 rows.")
