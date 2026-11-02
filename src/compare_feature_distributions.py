from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = BASE_DIR / "data" / "url_features_clean.csv"
EXTERNAL_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"

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
    original = pd.read_csv(ORIGINAL_PATH)
    external = pd.read_csv(EXTERNAL_PATH)

    print("\n--- EXTRACTING EXTERNAL FEATURES ---")

    from url_features import extract_url_features

    external_features = pd.DataFrame(
        [
            extract_url_features(str(url))
            for url in external["url"]
        ]
    )

    print("\n--- FEATURE DISTRIBUTION COMPARISON ---")

    for feature in FEATURES:
        original_mean = original[feature].mean()
        external_mean = external_features[feature].mean()

        print(f"\nFeature: {feature}")
        print(f"Original mean: {original_mean:.3f}")
        print(f"External mean: {external_mean:.3f}")

    print("\nComparison completed.")
    print("No URLs were opened or visited.")


if __name__ == "__main__":
    main()