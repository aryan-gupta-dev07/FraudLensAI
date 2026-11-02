from pathlib import Path
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"]).copy()
y = df["label"].astype(int)

print("Original feature count:", X.shape[1])

engineered = X.copy()

engineered["hostname_url_ratio"] = (
    engineered["hostname_length"] /
    engineered["url_length"].clip(lower=1)
)

engineered["path_url_ratio"] = (
    engineered["path_length"] /
    engineered["url_length"].clip(lower=1)
)

engineered["digit_url_ratio"] = (
    engineered["digit_count"] /
    engineered["url_length"].clip(lower=1)
)

engineered["special_char_url_ratio"] = (
    engineered["special_character_count"] /
    engineered["url_length"].clip(lower=1)
)

engineered["has_path"] = (
    engineered["path_length"] > 0
).astype(int)

print("Engineered feature count:", engineered.shape[1])


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

def evaluate_features(features, name):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    start = time.time()

    scores = cross_validate(
        model,
        features,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
        error_score="raise",
    )

    return {
        "Experiment": name,
        "Feature Count": features.shape[1],
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "Time (seconds)": round(time.time() - start, 2),
    }


print("\nEvaluating original features...")
original_result = evaluate_features(X, "Original Features")

print("\nEvaluating engineered features...")
engineered_result = evaluate_features(
    engineered,
    "Original + Engineered Features",
)

results = pd.DataFrame([
    original_result,
    engineered_result,
])

print("\n=== FEATURE ENGINEERING RESULTS ===")
print(results.to_string(index=False))

REPORT_PATH = (
    BASE_DIR / "reports" / "feature_engineering_comparison.csv"
)

results.to_csv(REPORT_PATH, index=False)

print("\nResults saved to:", REPORT_PATH)