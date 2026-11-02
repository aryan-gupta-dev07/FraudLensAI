from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudlens_url_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "baseline_model_evaluation.txt"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Loading original baseline model...")
model = joblib.load(MODEL_PATH)

start_time = time.time()
predictions = model.predict(X_test)
evaluation_time = time.time() - start_time

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, zero_division=0)
matrix = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = matrix.ravel()

output = f"""
=== BASELINE URL MODEL EVALUATION ===

Testing rows: {len(X_test)}
Accuracy: {accuracy:.6f}
Evaluation time: {evaluation_time:.2f} seconds

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

REPORT_PATH.write_text(output, encoding="utf-8")

print("Report saved to:", REPORT_PATH)
print("Original model was not modified.")