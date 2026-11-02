from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "independent" / "Dataset.csv"
)

df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

print(df["label"].value_counts(dropna=False))
print("\nLabel percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100).round(2)
)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isna().sum()[df.isna().sum() > 0])

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate URLs:", df["url"].duplicated().sum())

print("\n" + "=" * 60)
print("URL LABEL CONSISTENCY")
print("=" * 60)

label_counts = df.groupby("url")["label"].nunique()

conflicting_urls = label_counts[label_counts > 1]

print("URLs with conflicting labels:", len(conflicting_urls))

print("\n" + "=" * 60)
print("LABEL VALUES")
print("=" * 60)

print("Unique labels:", sorted(df["label"].dropna().unique()))

print("\nAnalysis completed. Original dataset unchanged.")