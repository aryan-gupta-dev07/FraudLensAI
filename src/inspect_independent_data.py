from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

files = [
    BASE_DIR / "data" / "independent" / "Dataset.csv",
    BASE_DIR / "data" / "independent" / "tranco.csv",
    BASE_DIR / "data" / "independent" / "tranco_clean.csv",
]

for file_path in files:
    print("\n" + "=" * 65)
    print("FILE:", file_path.name)

    if not file_path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(file_path, low_memory=False)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", list(df.columns))
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:", int(df.isna().sum().sum()))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\n" + "-" * 65)