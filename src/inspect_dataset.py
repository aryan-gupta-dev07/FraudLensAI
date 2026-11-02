from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "urlset.csv"

df = pd.read_csv(DATA_PATH,
                 encoding="latin-1", low_memory=False, on_bad_lines="warn")

print("=== DATASET QUALITY REPORT ===")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\n=== LABEL DISTRIBUTION ===")
print(df["label"].value_counts(dropna=False).sort_index())

print("\n=== LABEL PERCENTAGES ===")
print(
    (df["label"].value_counts(normalize=True, dropna=False) * 100)
    .round(2)
    .sort_index()
)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== DOMAIN EXAMPLES BY LABEL ===")
for label in sorted(df["label"].dropna().unique()):
    print(f"\nLabel {label}:")
    print(df.loc[df["label"] == label, "domain"].head(3).to_string(index=False))