from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

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


def main():
    print("Loading external dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Loading existing FraudLens model...")
    model = joblib.load(MODEL_PATH)

    feature_rows = []
    labels = []

    print("Extracting URL features...")

    skipped = 0

    for _, row in df.iterrows():
        try:
            features = extract_url_features(str(row["url"]))
            feature_rows.append(features)
            labels.append(int(row["label"]))
        except (ValueError, TypeError, AttributeError):
            skipped += 1

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    y = pd.Series(labels, name="actual_label")

    print("\nGenerating predictions...")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    phishing_column = list(model.classes_).index(1)
    phishing_scores = probabilities[:, phishing_column]

    results = pd.DataFrame({
        "actual_label": y,
        "predicted_label": predictions,
        "phishing_score": phishing_scores,
    })

    print("\n=== PREDICTION SUMMARY ===")
    print("Valid URLs:", len(results))
    print("Skipped URLs:", skipped)

    print("\nActual label distribution:")
    print(results["actual_label"].value_counts().sort_index())

    print("\nPredicted label distribution:")
    print(results["predicted_label"].value_counts().sort_index())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        target_names=["Benign", "Phishing"],
        zero_division=0,
    ))

    print("\nAverage phishing score by actual label:")
    print(results.groupby("actual_label")["phishing_score"].mean().round(4))

    print("\nAverage phishing score by actual and predicted label:")
    print(
        results.groupby(
            ["actual_label", "predicted_label"]
        )["phishing_score"].agg(["count", "mean", "median"]).round(4)
    )

    print("\nOriginal model was not modified.")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()