import re
import ipaddress

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
        return result.top_domain_under_public_suffix or hostname


data = pd.read_csv(DATA_PATH)
raw = pd.read_csv(RAW_DATA_PATH)

y = data["label"]
groups = raw["domain"].astype(str).apply(get_registered_domain)

# Same original 10 features
baseline_features = [
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

X_baseline = data[baseline_features]
X_advanced = data.drop(columns=["label"])

# One identical domain-separated split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X_advanced, y, groups=groups)
)


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


def evaluate(name, features):
    model = create_model()

    model.fit(
        features.iloc[train_idx],
        y.iloc[train_idx],
    )

    predictions = model.predict(features.iloc[test_idx])
    actual = y.iloc[test_idx]

    results = {
        "Accuracy": accuracy_score(actual, predictions),
        "Precision": precision_score(
            actual, predictions, zero_division=0
        ),
        "Recall": recall_score(
            actual, predictions, zero_division=0
        ),
        "F1": f1_score(actual, predictions, zero_division=0),
    }

    print(f"\n{name}")
    print("-" * 35)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(actual, predictions))

    return results


assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[test_idx])
)

print("Training rows:", len(train_idx))
print("Validation rows:", len(test_idx))
print("Domain overlap: 0")

baseline_results = evaluate(
    "BASELINE MODEL (10 FEATURES)",
    X_baseline,
)

advanced_results = evaluate(
    "ADVANCED MODEL (19 FEATURES)",
    X_advanced,
)

print("\nADVANCED MINUS BASELINE")
print("-" * 35)

for metric in baseline_results:
    difference = advanced_results[metric] - baseline_results[metric]
    print(f"{metric} difference: {difference:+.4f}")