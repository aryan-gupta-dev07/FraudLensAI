from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Give identical feature patterns the same group ID.
groups = pd.util.hash_pandas_object(
    X,
    index=False,
).to_numpy()

# Split entire feature-pattern groups, not individual rows.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

print("Training model with grouped split...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== GROUP-SPLIT EVALUATION ===")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])

print(
    "\nShared feature-pattern groups:",
    len(train_groups.intersection(test_groups)),
)

print("\nEvaluation completed. Existing model was not overwritten.")