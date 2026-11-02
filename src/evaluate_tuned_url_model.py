from pathlib import Path
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
REPORT_PATH = BASE_DIR / "reports" / "tuned_model_evaluation.txt"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

print("Training tuned Random Forest...")

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)
matrix = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = matrix.ravel()

output = f"""
=== TUNED URL MODEL EVALUATION ===

Training rows: {len(X_train)}
Testing rows: {len(X_test)}

Accuracy: {accuracy:.6f}
Training time: {training_time:.2f} seconds

Classification report:
{report}

Confusion matrix:
{matrix}

True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives: {tp}
"""

print(output)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(output, encoding="utf-8")

print("Evaluation saved to:", REPORT_PATH)
print("Original model was not modified.")