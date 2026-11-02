from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
REPORT_PATH = BASE_DIR / "reports" / "group_stability.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

groups = pd.util.hash_pandas_object(
    X,
    index=False,
).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=5,
    test_size=0.20,
    random_state=42,
)

results = []

for split_number, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1,
):
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

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "split": split_number,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "class_1_test_share": y_test.mean(),
        "accuracy": accuracy_score(y_test, predictions),
        "class_1_precision": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "class_1_recall": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
    })

results_df = pd.DataFrame(results)

print("=== GROUPED SPLIT STABILITY ===")
print(results_df.round(4).to_string(index=False))

print("\n=== MEAN METRICS ACROSS FIVE SPLITS ===")
print(results_df[
    [
        "class_1_test_share",
        "accuracy",
        "class_1_precision",
        "class_1_recall",
    ]
].mean().round(4).to_string())

print("\n=== METRIC STANDARD DEVIATION ===")
print(results_df[
    [
        "accuracy",
        "class_1_precision",
        "class_1_recall",
    ]
].std().round(4).to_string())

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_PATH, index=False)

print("\nReport saved to:", REPORT_PATH)
print("Existing production model was not overwritten.")