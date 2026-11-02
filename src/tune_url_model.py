from pathlib import Path
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=False,
    return_train_score=False,
)

print("Starting Random Forest hyperparameter tuning...")

start_time = time.time()
search.fit(X, y)
elapsed = time.time() - start_time

results = pd.DataFrame(search.cv_results_)

columns = [
    "param_classifier__n_estimators",
    "param_classifier__max_depth",
    "param_classifier__min_samples_split",
    "param_classifier__min_samples_leaf",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]

results = results[columns].sort_values("rank_test_score")

print("\n=== TOP 10 HYPERPARAMETER CONFIGURATIONS ===")
print(results.head(10).to_string(index=False))

print("\nBest parameters:")
print(search.best_params_)

print(f"\nBest mean CV F1: {search.best_score_:.6f}")
print(f"Total time: {elapsed:.2f} seconds")

REPORT_PATH = BASE_DIR / "reports" / "hyperparameter_tuning.csv"
results.to_csv(REPORT_PATH, index=False)

print("\nResults saved to:", REPORT_PATH)