from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(DATA_PATH)

print("=== TRAINING DATA LABEL DISTRIBUTION ===")
print(df["label"].value_counts())

print("\n=== DUPLICATE FEATURE ROWS ===")

features = df.drop(columns=["label"])

print("Total rows:", len(df))
print("Unique feature combinations:", len(features.drop_duplicates()))

print("\n=== EXAMPLE.COM-LIKE FEATURE PATTERN ===")

sample = {
    "url_length": 19,
    "hostname_length": 11,
    "path_length": 0,
    "subdomain_count": 0,
    "uses_ip_address": 0,
    "contains_at_symbol": 0,
    "contains_dash_in_hostname": 0,
    "https_scheme": 1,
    "digit_count": 0,
    "special_character_count": 4,
}

matches = df[
    (df[list(sample)] == pd.Series(sample)).all(axis=1)
]

print("Matching rows:", len(matches))

if not matches.empty:
    print("Label distribution for matching rows:")
    print(matches["label"].value_counts())
else:
    print("No exact feature matches found.")