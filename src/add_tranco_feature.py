import pandas as pd
import tldextract
from pathlib import Path

DATA = Path("data")
extractor = tldextract.TLDExtract(suffix_list_urls=())

# Load existing training data
urls = pd.read_csv(DATA / "urlset_clean.csv")
features = pd.read_csv(DATA / "advanced_url_features.csv")

# Load Tranco rankings
tranco = pd.read_csv(
    DATA / "independent" / "tranco_clean.csv"
)

rank_map = dict(zip(tranco["domain"], tranco["rank"]))

def get_registered_domain(value):
    value = str(value).strip()

    if "://" not in value:
        value = "https://" + value

    result = extractor(value)

    return result.top_domain_under_public_suffix

# Extract registered domains
domains = urls["domain"].map(get_registered_domain)

# Add Tranco rank; 0 means not present in the downloaded list
features["tranco_rank"] = domains.map(rank_map).fillna(0)

# Save as a separate experimental dataset
output = DATA / "advanced_url_features_tranco.csv"
features.to_csv(output, index=False)

print("Tranco feature integration complete")
print("Rows:", len(features))
print("Features:", len(features.columns) - 1)
print("Matched:", int((features["tranco_rank"] > 0).sum()))
print("Missing:", int((features["tranco_rank"] == 0).sum()))
print("Saved:", output)