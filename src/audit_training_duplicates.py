import pandas as pd
from pathlib import Path


# Load the original training dataset
DATA_PATH = Path("data/urlset_clean.csv")

df = pd.read_csv(DATA_PATH)

# Normalize URLs conservatively
df["normalized_domain"] = (
    df["domain"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Basic URL statistics
total_records = len(df)
unique_urls = df["normalized_domain"].nunique()

# Identify duplicate URL groups
duplicate_mask = df["normalized_domain"].duplicated(keep=False)
duplicate_records = df[duplicate_mask]

duplicate_groups = (
    duplicate_records
    .groupby("normalized_domain")
)

# Count groups containing conflicting labels
conflicting_groups = 0

for _, group in duplicate_groups:
    if group["label"].nunique() > 1:
        conflicting_groups += 1

# Calculate duplicate excess records
duplicate_excess = total_records - unique_urls

print("=" * 55)
print("FRAUDLENS AI - TRAINING DUPLICATE AUDIT")
print("=" * 55)

print(f"Total records: {total_records}")
print(f"Unique normalized URLs: {unique_urls}")
print(f"Duplicate excess records: {duplicate_excess}")
print(f"Records belonging to duplicate groups: {len(duplicate_records)}")
print(f"Duplicate URL groups: {len(duplicate_groups)}")
print(f"Groups with conflicting labels: {conflicting_groups}")

print("\nSample duplicate URLs:")

if len(duplicate_records) > 0:
    print(
        duplicate_records[
            ["domain", "label"]
        ].head(10).to_string(index=False)
    )
else:
    print("No duplicate URLs found.")

print("\nAudit completed.")