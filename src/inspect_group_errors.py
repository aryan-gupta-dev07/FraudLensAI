from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
REPORT_PATH = BASE_DIR / "reports" / "group_error_analysis.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Keep identical feature patterns in the same group.
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

print("Training grouped evaluation model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Identify incorrectly classified test rows.
errors = X_test.copy()
errors["actual_label"] = y_test
errors["predicted_label"] = predictions

errors = errors[
    errors["actual_label"] != errors["predicted_label"]
].copy()

errors["error_type"] = errors.apply(
    lambda row: (
        "Missed suspicious example"
        if row["actual_label"] == 1
        else "False alarm on class 0"
    ),
    axis=1,
)

print("\n=== GROUPED MODEL ERROR ANALYSIS ===")
print("Test rows:", len(X_test))
print("Incorrect predictions:", len(errors))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nError types:")
print(errors["error_type"].value_counts())

print("\nFeature averages by error type:")
if not errors.empty:
    print(
        errors.groupby("error_type")[
            list(X.columns)
        ].mean().round(2).to_string()
    )

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
errors.to_csv(REPORT_PATH, index=False)

print("\nError report saved to:", REPORT_PATH)
print("Existing production model was not overwritten.")