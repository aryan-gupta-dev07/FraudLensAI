import re
import ipaddress
import joblib
import pandas as pd
import tldextract

from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


DATA_PATH = "data/advanced_url_features.csv"
RAW_DATA_PATH = "data/urlset_clean.csv"
MODEL_PATH = "models/fraudlens_advanced_model.pkl"

extractor = tldextract.TLDExtract(suffix_list_urls=())


def get_registered_domain(url):
    candidate = (
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
    )

    candidate = re.sub(r"\s", "%20", candidate)
    hostname = urlparse(candidate).hostname

    if not hostname:
        return "unknown"

    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        result = extractor(hostname)
        return result.registered_domain or hostname


# Load datasets
data = pd.read_csv(DATA_PATH)
raw = pd.read_csv(RAW_DATA_PATH)

X = data.drop(columns=["label"])
y = data["label"]

# Group URLs by registered domain
groups = raw["domain"].astype(str).apply(get_registered_domain)

# Group-aware train-validation split
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

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(test_groups)

# Train new model
model = Pipeline([
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

print("Training advanced model...")

model.fit(X_train, y_train)

# Evaluate new model
predictions = model.predict(X_test)

print("\nADVANCED MODEL RESULTS")
print("----------------------")
print("Training rows:", len(train_idx))
print("Validation rows:", len(test_idx))
print("Training domains:", len(train_groups))
print("Validation domains:", len(test_groups))
print("Domain overlap:", len(train_groups & test_groups))

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1 score:", round(f1_score(y_test, predictions, zero_division=0), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save separately; original model remains untouched
joblib.dump(model, MODEL_PATH)

print("\nNew model saved:", MODEL_PATH)