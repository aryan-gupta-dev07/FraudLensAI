from pathlib import Path

import pandas as pd

from url_features import extract_url_features


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "data" / "independent" / "Dataset.csv"
)

FEATURE_MAPPING = {
    "url_length": "url_len",
    "hostname_length": "dom_len",
    "path_length": "path_len",
    "subdomain_count": "subdom_cnt",
    "uses_ip_address": "is_ip",
    "https_scheme": "is_https",
    "digit_count": "digit_cnt",
    "special_character_count": "special_cnt",
}


def main():
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    print("=" * 65)
    print("EXTERNAL FEATURE COMPATIBILITY AUDIT")
    print("=" * 65)

    print(f"\nTotal external records: {len(df)}")

    extracted_features = []
    failed_rows = []

    for index, url in enumerate(df["url"]):
        try:
            features = extract_url_features(str(url))
            extracted_features.append(features)

        except (ValueError, TypeError, AttributeError):
            extracted_features.append(None)
            failed_rows.append(index)

    valid_indices = [
        index
        for index, features in enumerate(extracted_features)
        if features is not None
    ]

    print(f"Successfully extracted: {len(valid_indices)}")
    print(f"Failed extractions: {len(failed_rows)}")

    print("\nFEATURE-WISE COMPARISON")
    print("-" * 65)

    for original_feature, external_column in FEATURE_MAPPING.items():

        original_values = [
            extracted_features[index][original_feature]
            for index in valid_indices
        ]

        external_values = (
            df.iloc[valid_indices][external_column]
            .reset_index(drop=True)
        )

        original_series = pd.Series(original_values)

        matches = (
            original_series == external_values
        )

        match_count = int(matches.sum())
        total = len(matches)

        percentage = (
            match_count / total * 100 if total else 0
        )

        print(f"\nFeature: {original_feature}")
        print(f"External column: {external_column}")
        print(f"Exact matches: {match_count}/{total}")
        print(f"Match percentage: {percentage:.2f}%")

        mismatch_indices = matches[~matches].index[:3]

        if len(mismatch_indices) > 0:
            print("Mismatch examples:")

            for position in mismatch_indices:
                actual_index = valid_indices[position]

                print(
                    f"  URL: {df.iloc[actual_index]['url']}"
                )
                print(
                    f"  Extractor: {original_values[position]}"
                )
                print(
                    f"  External: {external_values.iloc[position]}"
                )


if __name__ == "__main__":
    main()