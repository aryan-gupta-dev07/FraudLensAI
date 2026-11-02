from pathlib import Path

import pandas as pd

from url_features import extract_url_features


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "subdomain_count",
    "https_scheme",
    "digit_count",
    "special_character_count",
]


def main():
    df = pd.read_csv(DATA_PATH)

    rows = []
    skipped = 0

    print("Extracting external URL features...")

    for _, row in df.iterrows():
        try:
            features = extract_url_features(str(row["url"]))
            features["label"] = int(row["label"])
            rows.append(features)

        except (ValueError, TypeError, AttributeError):
            skipped += 1

    result = pd.DataFrame(rows)

    print("\nValid URLs:", len(result))
    print("Skipped URLs:", skipped)

    print("\nEXTERNAL FEATURE MEANS BY LABEL")

    print(
        result.groupby("label")[FEATURE_COLUMNS]
        .mean()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()