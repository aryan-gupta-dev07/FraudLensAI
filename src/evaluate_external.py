import re
import ipaddress

import joblib
import pandas as pd
import tldextract

from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.advanced_url_features import extract_advanced_url_features


TRAIN_PATH = "data/advanced_url_features.csv"
EXTERNAL_PATH = "data/independent/Dataset.csv"

BASELINE_MODEL_PATH = "models/fraudlens_external_baseline.pkl"
ADVANCED_MODEL_PATH = "models/fraudlens_external_advanced.pkl"

BASELINE_FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "subdomain_count",
    "uses_ip_address",
    "contains_at_symbol",
    "contains_dash_in_hostname",
    "https_scheme",
    "digit_count",
    "special_character_count",
]

# Load datasets
train = pd.read_csv(TRAIN_PATH)
external = pd.read_csv(EXTERNAL_PATH)

# Identify URL column
url_candidates = ["url", "domain", "website", "link"]
url_column = next(
    (c for c in external.columns if c.lower() in url_candidates),
    None,
)

if url_column is None:
    raise ValueError(
        f"URL column not found. Available columns: {external.columns.tolist()}"
    )

if "label" not in external.columns:
    raise ValueError("External dataset has no label column.")

print("External URL column:", url_column)
print("External rows:", len(external))

# Prepare external features locally.
# No URLs are opened or visited.
feature_rows = []

for url in external[url_column]:
    try:
        feature_rows.append(
            extract_advanced_url_features(str(url))
        )
    except (ValueError, TypeError):
        feature_rows.append(None)

valid_indices = [
    i for i, row in enumerate(feature_rows)
    if row is not None
]

if not valid_indices:
    raise ValueError("No valid external URLs found.")

X_external = pd.DataFrame(
    [feature_rows[i] for i in valid_indices]
)

y_external = external["label"].iloc[valid_indices].astype(int)

print("Valid external URLs:", len(valid_indices))
print("Skipped URLs:", len(external) - len(valid_indices))
print("External labels:", y_external.value_counts().to_dict())

# Training features
X_train_advanced = train.drop(columns=["label"])
y_train = train["label"].astype(int)

X_train_baseline = X_train_advanced[BASELINE_FEATURES]

# Create identical model configurations
def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])


def evaluate_model(name, model, X, y):
    predictions = model.predict(X)

    print(f"\n{name}")
    print("-" * 40)

    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print(
        "Precision:",
        round(precision_score(y, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y, predictions, zero_division=0), 4),
    )
    print("F1:", round(f1_score(y, predictions, zero_division=0), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    return predictions


# Train baseline model
print("\nTraining baseline model...")

baseline_model = create_model()
baseline_model.fit(X_train_baseline, y_train)

joblib.dump(baseline_model, BASELINE_MODEL_PATH)

# Train advanced model
print("\nTraining advanced model...")

advanced_model = create_model()
advanced_model.fit(X_train_advanced, y_train)

joblib.dump(advanced_model, ADVANCED_MODEL_PATH)

# Evaluate both on exactly the same external records
X_external_baseline = X_external[BASELINE_FEATURES]

baseline_predictions = evaluate_model(
    "BASELINE MODEL - EXTERNAL DATA",
    baseline_model,
    X_external_baseline,
    y_external,
)

advanced_predictions = evaluate_model(
    "ADVANCED MODEL - EXTERNAL DATA",
    advanced_model,
    X_external,
    y_external,
)

print("\nEXTERNAL EVALUATION COMPLETE")
print("Baseline model:", BASELINE_MODEL_PATH)
print("Advanced model:", ADVANCED_MODEL_PATH)