import os
import json
import joblib
import numpy as np
import pandas as pd
import tldextract

from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "reports/external_reextracted_features.csv"
URL_PATH = "data/independent/Dataset.csv"

MODEL_PATH = "models/fraudlens_domain_adapted_model.pkl"
REPORT_PATH = "reports/domain_adapted_evaluation.json"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and verify data
# --------------------------------------------------

features = pd.read_csv(DATA_PATH)
original = pd.read_csv(URL_PATH)

if len(features) != len(original):
    raise ValueError("Feature and URL row counts do not match.")

if not np.array_equal(
    features["label"].to_numpy(),
    original["label"].to_numpy()
):
    raise ValueError("Feature and URL labels do not match.")

feature_columns = [
    column for column in features.columns
    if column != "label"
]

X = features[feature_columns].copy()
y = features["label"].astype(int)

if X.isna().any().any():
    raise ValueError("Missing feature values detected.")

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

# --------------------------------------------------
# 3. Extract domain groups
# --------------------------------------------------

extractor = tldextract.TLDExtract(suffix_list_urls=())

def get_domain(url):
    hostname = urlparse(str(url)).hostname or ""
    hostname = hostname.lower().strip(".")

    result = extractor(hostname)

    if result.domain and result.suffix:
        return result.domain + "." + result.suffix

    return hostname if hostname else str(url)

groups = original["url"].apply(get_domain)

print("Total rows:", len(X))
print("Unique domain groups:", groups.nunique())
print("Class distribution:", y.value_counts().to_dict())

# --------------------------------------------------
# 4. Domain-separated 70/15/15 split
# --------------------------------------------------

first_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=RANDOM_STATE
)

train_val_idx, test_idx = next(
    first_split.split(X, y, groups)
)

X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
g_train_val = groups.iloc[train_val_idx]

second_split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.1765,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    second_split.split(
        X_train_val,
        y_train_val,
        g_train_val
    )
)

X_train = X_train_val.iloc[train_idx]
y_train = y_train_val.iloc[train_idx]

X_val = X_train_val.iloc[val_idx]
y_val = y_train_val.iloc[val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

train_domains = set(g_train_val.iloc[train_idx])
val_domains = set(g_train_val.iloc[val_idx])
test_domains = set(groups.iloc[test_idx])

# Verify no domain leakage
assert train_domains.isdisjoint(val_domains)
assert train_domains.isdisjoint(test_domains)
assert val_domains.isdisjoint(test_domains)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("\nTraining ExtraTrees model...")

model = ExtraTreesClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")

# --------------------------------------------------
# 6. Select threshold on validation only
# --------------------------------------------------

val_scores = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)

valid_candidates = []

for threshold in thresholds:
    predictions = (val_scores >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    if recall >= 0.80:
        valid_candidates.append(
            (f1, threshold, precision, recall)
        )

if valid_candidates:
    best = max(valid_candidates, key=lambda item: item[0])
else:
    raise RuntimeError(
        "No validation threshold achieved recall >= 0.80."
    )

best_f1, best_threshold, best_precision, best_recall = best

print("\nSelected validation threshold:", best_threshold)
print("Validation F1:", best_f1)
print("Validation precision:", best_precision)
print("Validation recall:", best_recall)

# --------------------------------------------------
# 7. Final untouched test evaluation
# --------------------------------------------------

test_scores = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_scores >= best_threshold
).astype(int)

results = {
    "dataset": "Mendeley external URL dataset",
    "split_method": "Domain-separated GroupShuffleSplit",
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_scores)),
    "pr_auc": float(average_precision_score(y_test, test_scores)),
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        test_predictions,
        zero_division=0,
        output_dict=True
    ),
    "domain_overlap": False
}

# --------------------------------------------------
# 8. Save artifacts
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

with open(
    "models/fraudlens_domain_adapted_threshold.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {"threshold": float(best_threshold)},
        file,
        indent=4
    )

# --------------------------------------------------
# 9. Display final results
# --------------------------------------------------

print("\n========== FINAL TEST RESULTS ==========")

for key, value in results.items():
    if key != "classification_report":
        print(f"{key}: {value}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
print("\nTraining and evaluation completed.")