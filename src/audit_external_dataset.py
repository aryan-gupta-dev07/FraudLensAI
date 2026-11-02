from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Original Feature Dataset": (
        BASE_DIR / "data" / "url_features_clean.csv"
    ),
    "Independent External Dataset": (
        BASE_DIR / "data" / "independent" / "Dataset.csv"
    ),
}


def audit_dataset(name, file_path):
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path, low_memory=False)

    print(f"\nFile: {file_path.name}")
    print(f"Total records: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f" - {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string()
          if missing.any()
          else "No missing values")

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nFirst 5 rows (first 6 columns only):")
    print(df.iloc[:5, :6].to_string(index=False))

    print("\nPotential label columns:")
    for column in df.columns:
        if any(word in column.lower()
               for word in ["label", "class", "target", "result"]):
            print(f"\nColumn: {column}")
            print(df[column].value_counts(dropna=False).to_string())


def main():
    for name, file_path in DATASETS.items():
        audit_dataset(name, file_path)


if __name__ == "__main__":
    main()