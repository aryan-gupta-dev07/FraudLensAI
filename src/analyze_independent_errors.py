from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

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
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    print("Extracting features locally...")

    X = pd.DataFrame(
        [
            extract_url_features(str(url))
            for url in df["url"]
        ],
        columns=FEATURE_COLUMNS,
    )

    y = df["label"].astype(int).reset_index(drop=True)

    predictions = model.predict(X)

    results = X.copy()
    results["actual"] = y
    results["predicted"] = predictions

    false_positives = results[
        (results["actual"] == 0) &
        (results["predicted"] == 1)
    ]

    false_negatives = results[
        (results["actual"] == 1) &
        (results["predicted"] == 0)
    ]

    print("\n--- ERROR COUNTS ---")
    print("False Positives:", len(false_positives))
    print("False Negatives:", len(false_negatives))

    print("\n--- FALSE POSITIVE FEATURE MEANS ---")
    print(
        false_positives[FEATURE_COLUMNS]
        .mean()
        .round(3)
        .to_string()
    )

    print("\n--- FALSE NEGATIVE FEATURE MEANS ---")
    print(
        false_negatives[FEATURE_COLUMNS]
        .mean()
        .round(3)
        .to_string()
    )

    report_path = BASE_DIR / "reports" / "independent_error_analysis.csv"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "error_type": ["False Positive", "False Negative"],
        "count": [len(false_positives), len(false_negatives)],
    })

    summary.to_csv(report_path, index=False)

    print("\nError summary saved to:", report_path)
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()