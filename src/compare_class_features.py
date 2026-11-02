from pathlib import Path

import pandas as pd

from url_features import extract_url_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"

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

    print("Extracting features locally...")

    extracted = pd.DataFrame(
        [extract_url_features(str(url)) for url in df["url"]]
    )

    extracted["label"] = df["label"].values

    print("\n--- CLASS-WISE FEATURE MEANS ---")

    result = extracted.groupby("label")[FEATURES].mean().T

    result.columns = ["Benign", "Phishing"]

    print(result.round(3).to_string())

    print("\nLabel mapping: 0 = Benign, 1 = Phishing")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()