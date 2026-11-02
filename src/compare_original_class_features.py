from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"

FEATURES = [
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

    print("--- ORIGINAL DATASET CLASS-WISE MEANS ---")

    result = df.groupby("label")[FEATURES].mean().T

    result.columns = ["Benign", "Phishing"]

    print(result.round(3).to_string())

    print("\nLabel mapping: 0 = Benign, 1 = Phishing")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()