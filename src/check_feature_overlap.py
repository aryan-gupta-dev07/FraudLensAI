from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

print("Loading URL feature dataset...")

df = pd.read_csv(DATA_PATH)

# Separate features and labels.
X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Reproduce the existing training/testing split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Include the label to check whether identical feature patterns
# appear with different labels.
feature_columns = list(X.columns)

unique_patterns = df.drop_duplicates(
    subset=feature_columns
)

patterns_with_conflicting_labels = (
    df.groupby(feature_columns, dropna=False)["label"]
    .nunique()
)

conflicting_pattern_count = int(
    (patterns_with_conflicting_labels > 1).sum()
)

# Count repeated feature patterns in the full dataset.
repeated_pattern_rows = int(
    df.duplicated(subset=feature_columns, keep=False).sum()
)

# Check feature-pattern overlap between training and testing.
train_patterns = set(
    map(tuple, X_train[feature_columns].to_numpy())
)

test_patterns = set(
    map(tuple, X_test[feature_columns].to_numpy())
)

overlap_patterns = train_patterns.intersection(test_patterns)

print("\n=== DATASET OVERLAP REPORT ===")
print(f"Total rows: {len(df)}")
print(f"Feature columns: {len(feature_columns)}")
print(f"Unique feature patterns: {len(unique_patterns)}")
print(f"Rows belonging to repeated patterns: {repeated_pattern_rows}")
print(
    "Patterns associated with conflicting labels:",
    conflicting_pattern_count,
)
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Patterns shared between train and test: {len(overlap_patterns)}")

if overlap_patterns:
    print(
        "\nWARNING: Identical feature patterns occur in both "
        "training and testing sets."
    )
else:
    print(
        "\nNo identical feature patterns were found across "
        "the training and testing sets."
    )

print("\nDataset overlap check completed.")