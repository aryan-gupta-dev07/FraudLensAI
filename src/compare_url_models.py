from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42,
    ),
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

results = []

for name, classifier in models.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])

    print(f"\nEvaluating {name}...")
    start_time = time.time()

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    elapsed = time.time() - start_time

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "Training and Evaluation Time (seconds)": round(elapsed, 2),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1", ascending=False)

print("\n=== MODEL COMPARISON RESULTS ===")
print(results_df.to_string(index=False))

REPORT_PATH = BASE_DIR / "reports" / "model_comparison.csv"
results_df.to_csv(REPORT_PATH, index=False)

print("\nResults saved to:", REPORT_PATH)