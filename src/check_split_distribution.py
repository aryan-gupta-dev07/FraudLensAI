from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

groups = pd.util.hash_pandas_object(
    X,
    index=False,
).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("=== SPLIT DISTRIBUTION CHECK ===")

print("\nFull dataset label distribution:")
print(y.value_counts().sort_index())

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())

print("\nClass proportions:")
print("Full dataset:")
print(y.value_counts(normalize=True).sort_index().round(4))

print("\nTraining:")
print(y_train.value_counts(normalize=True).sort_index().round(4))

print("\nTesting:")
print(y_test.value_counts(normalize=True).sort_index().round(4))