from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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


def main():
    print("Loading independent dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Loading existing FraudLens model...")
    model = joblib.load(MODEL_PATH)

    print("\nExtracting URL features locally...")

    feature_rows = []
    valid_indices = []

    for index, url in df["url"].items():
        try:
            features = extract_url_features(str(url))
            feature_rows.append(features)
            valid_indices.append(index)
        except (ValueError, TypeError, AttributeError):
            continue

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

    y = (
        df.loc[valid_indices, "label"]
        .astype(int)
        .reset_index(drop=True)
    )

    print("Valid URLs:", len(X))
    print("Skipped URLs:", len(df) - len(X))

    print("\nGenerating predictions...")
    predictions = model.predict(X)

    print("\n=== INDEPENDENT DATASET EVALUATION ===")
    print("Evaluation rows:", len(X))
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Benign", "Phishing"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))

    print("\nExisting model was not retrained or modified.")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()