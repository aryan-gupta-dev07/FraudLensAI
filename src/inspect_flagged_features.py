from pathlib import Path
import pandas as pd

from advanced_url_features import extract_advanced_url_features

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "reports" / "tranco_flagged_review.csv"
OUTPUT_PATH = BASE_DIR / "reports" / "tranco_flagged_features.csv"

df = pd.read_csv(INPUT_PATH)

feature_rows = []

for _, row in df.iterrows():
    domain = str(row["domain"])

    features = extract_advanced_url_features(
        "https://" + domain
    )

    feature_rows.append({
        "rank": row["rank"],
        "domain": domain,
        "risk_score": row["risk_score"],
        **features,
    })

result = pd.DataFrame(feature_rows)

result.to_csv(OUTPUT_PATH, index=False)

print("Flagged domains inspected:", len(result))
print("Feature report saved to:", OUTPUT_PATH)

print("\nTop 10 domains with selected features:")

columns = [
    "domain",
    "risk_score",
    "url_length",
    "hostname_length",
    "subdomain_count",
    "uses_ip_address",
    "https_scheme",
    "digit_count",
    "suspicious_token_count",
    "embedded_brand_count",
    "brand_mismatch",
]

print(
    result[columns]
    .sort_values("risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("FEATURE PATTERN SUMMARY")
print("=" * 60)

numeric_columns = [
    "url_length",
    "hostname_length",
    "digit_count",
    "special_character_count",
    "hostname_token_count",
    "path_token_count",
    "suspicious_token_count",
    "embedded_domain_count",
    "embedded_brand_count",
]

available_columns = [
    col for col in numeric_columns if col in result.columns
]

print(
    result[available_columns]
    .describe()
    .round(2)
    .to_string()
)

print("\nPunycode domains:")
print(
    result["domain"]
    .str.startswith("xn--")
    .sum()
)

print("\nDomains containing digits:")
print(
    result["domain"]
    .str.contains(r"\d", regex=True)
    .sum()
)