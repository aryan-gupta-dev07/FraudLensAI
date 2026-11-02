import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

extractor = tldextract.TLDExtract(suffix_list_urls=())

raw = pd.read_csv("data/urlset_clean.csv")
groups = raw["domain"].astype(str).map(
    lambda u: extractor(
        u if "://" in u else "https://" + u
    ).top_domain_under_public_suffix
)

old = pd.read_csv("data/advanced_url_features.csv")
new = pd.read_csv("data/advanced_url_features_tranco.csv")

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(old, old["label"], groups)
)

def evaluate(data, name):
    X = data.drop(columns=["label"])
    y = data["label"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y.iloc[test_idx], pred), 4))
    print("Precision:", round(precision_score(y.iloc[test_idx], pred), 4))
    print("Recall:", round(recall_score(y.iloc[test_idx], pred), 4))
    print("F1:", round(f1_score(y.iloc[test_idx], pred), 4))

evaluate(old, "ORIGINAL 19 FEATURES")
evaluate(new, "TRANCO 20 FEATURES")