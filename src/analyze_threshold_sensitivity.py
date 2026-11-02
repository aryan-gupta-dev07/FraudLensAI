from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from url_features import extract_url_features


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudlens_url_model.pkl"

FEATURE_COLUMNS = [
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

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def main():
    print("Loading external dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Loading existing FraudLens model...")
    model = joblib.load(MODEL_PATH)

    feature_rows = []
    labels = []
    skipped = 0

    print("Extracting URL features...")

    for _, row in df.iterrows():
        try:
            features = extract_url_features(str(row["url"]))
            feature_rows.append(features)
            labels.append(int(row["label"]))
        except (ValueError, TypeError, AttributeError):
            skipped += 1

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = pd.Series(labels)

    print("\nGenerating phishing scores...")
    probabilities = model.predict_proba(X)

    phishing_column = list(model.classes_).index(1)
    phishing_scores = probabilities[:, phishing_column]

    results = []

    print("\nTesting different thresholds...")

    for threshold in THRESHOLDS:
        predictions = (phishing_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y, predictions),
            "precision": precision_score(
                y, predictions, zero_division=0
            ),
            "recall": recall_score(
                y, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y, predictions, zero_division=0
            ),
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "true_positives": tp,
        })

    report = pd.DataFrame(results)

    print("\n=== THRESHOLD SENSITIVITY RESULTS ===")
    print(report.round(4).to_string(index=False))

    output_path = BASE_DIR / "reports" / "threshold_sensitivity.csv"
    report.to_csv(output_path, index=False)

    print("\nValid URLs:", len(X))
    print("Skipped URLs:", skipped)
    print("Report saved to:", output_path)
    print("\nOriginal model was not modified.")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()