from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent.parent
input_file = base / "data" / "urlset.csv"
output_file = base / "data" / "urlset_clean.csv"

df = pd.read_csv(
    input_file,
    encoding="latin-1",
    on_bad_lines="skip",
    low_memory=False
)

df = df.dropna(subset=["domain", "label"])
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df = df[df["label"].isin([0, 1])]
df = df.drop_duplicates()

features = df.columns.drop(["domain", "label"])

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(output_file, index=False)

print("Cleaning complete")
print("Total rows:", len(df))
print("Label counts:")
print(df["label"].value_counts())
print("Saved to:", output_file)