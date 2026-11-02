from pathlib import Path

import joblib
import pandas as pd

from url_features import extract_url_features


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudlens_url_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    features = []

    for url in df["url"]:
        features.append(extract_url_features(str(url)))

    X = pd.DataFrame(features)

    predictions = model.predict(X)

    print("\n--- PREDICTION DISTRIBUTION ---")
    print(
        pd.Series(predictions)
        .map({0: "Benign", 1: "Phishing"})
        .value_counts()
    )

    print("\n--- ACTUAL DISTRIBUTION ---")
    print(
        df["label"]
        .map({0: "Benign", 1: "Phishing"})
        .value_counts()
    )

    print("\n--- FEATURE STATISTICS ---")
    print(X.describe().round(2).to_string())

    print("\nNo URLs were opened or visited.")


if __name__ == "__main__":
    main()