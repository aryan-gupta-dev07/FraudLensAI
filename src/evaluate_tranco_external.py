import pandas as pd
import tldextract

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

extractor = tldextract.TLDExtract(suffix_list_urls=())

# Load training data and Tranco rankings
train = pd.read_csv("data/advanced_url_features_tranco.csv")
tranco = pd.read_csv("data/independent/tranco_clean.csv")
rank_map = dict(zip(tranco["domain"], tranco["rank"]))

# Load independent external test data
external = pd.read_csv("data/independent/Dataset.csv")

def get_domain(url):
    value = str(url).strip()
    if "://" not in value:
        value = "https://" + value
    return extractor(value).top_domain_under_public_suffix

# Extract external URL features
rows = []

for url in external["url"]:
    features = extract_advanced_url_features(str(url))
    domain = get_domain(url)
    features["tranco_rank"] = rank_map.get(domain, 0)
    rows.append(features)

X_external = pd.DataFrame(rows)
y_external = external["label"]

X_train = train.drop(columns=["label"])
y_train = train["label"]

# Train experimental model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_external)

print("\nTRANCO MODEL - INDEPENDENT EXTERNAL TEST")
print("Accuracy:", round(accuracy_score(y_external, predictions), 4))
print("Precision:", round(precision_score(y_external, predictions), 4))
print("Recall:", round(recall_score(y_external, predictions), 4))
print("F1:", round(f1_score(y_external, predictions), 4))
print("Confusion matrix:")
print(confusion_matrix(y_external, predictions))