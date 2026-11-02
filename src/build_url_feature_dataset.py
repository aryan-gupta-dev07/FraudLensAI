from pathlib import Path

import pandas as pd

from url_features import extract_url_features

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "urlset_clean.csv"
OUTPUT_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(INPUT_PATH)

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["domain", "label"])
df = df[df["label"].isin([0, 1])]

records = []
skipped = 0

print("Extracting URL features...")

for _, row in df.iterrows():
    try:
        features = extract_url_features(str(row["domain"]))
        features["label"] = int(row["label"])
        records.append(features)
    except (ValueError, TypeError):
        skipped += 1

result = pd.DataFrame(records)


result.to_csv(OUTPUT_PATH, index=False)

print("\n=== FEATURE DATASET READY ===")
print("Rows saved:", len(result))
print("Feature count:", len(result.columns) - 1)
print("Skipped invalid URLs:", skipped)
print("\nLabel distribution:")
print(result["label"].value_counts().sort_index())
print("\nSaved to:", OUTPUT_PATH)