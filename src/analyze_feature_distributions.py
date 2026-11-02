from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_PATH = BASE_DIR / "data" / "url_features_clean.csv"
EXTERNAL_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"

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
    original = pd.read_csv(ORIGINAL_PATH)
    external = pd.read_csv(EXTERNAL_PATH, low_memory=False)

    results = []

    for original_feature, external_feature in FEATURE_MAPPING.items():

        original_values = original[original_feature].dropna()
        external_values = external[external_feature].dropna()

        results.append({
            "feature": original_feature,
            "original_mean": original_values.mean(),
            "external_mean": external_values.mean(),
            "original_median": original_values.median(),
            "external_median": external_values.median(),
            "original_std": original_values.std(),
            "external_std": external_values.std(),
            "original_missing": original[original_feature].isna().sum(),
            "external_missing": external[external_feature].isna().sum(),
        })

    report = pd.DataFrame(results)

    output_path = BASE_DIR / "reports" / "distribution_analysis.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(output_path, index=False)

    print("=" * 65)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 65)

    print(report.round(3).to_string(index=False))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()