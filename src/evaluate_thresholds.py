from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
REPORT_PATH = BASE_DIR / "reports" / "threshold_analysis.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

# Keep identical feature patterns together.
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

# Get the model's estimated scores for class 1.
class_labels = list(model.classes_)
class_1_index = class_labels.index(1)

scores = model.predict_proba(X_test)[:, class_1_index]

results = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append({
        "threshold": threshold,
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
        "false_alarms": int(fp),
        "missed_suspicious": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    })

results_df = pd.DataFrame(results)

print("\n=== THRESHOLD COMPARISON ===")
print(results_df.round(4).to_string(index=False))

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(REPORT_PATH, index=False)

print("\nReport saved to:", REPORT_PATH)
print("Existing production model was not overwritten.")