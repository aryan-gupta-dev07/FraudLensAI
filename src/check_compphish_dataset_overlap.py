from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

BASE = Path(__file__).resolve().parent.parent

FILES = {
    "CompPhish": (
        BASE / "data/independent/comp_phish/All_Features_threshold90.xlsx",
        "url",
    ),
    "Mendeley": (
        BASE / "data/independent/Dataset.csv",
        "url",
    ),
    "Original": (
        BASE / "data/urlset_clean.csv",
        "domain",
    ),
}


def normalize_url(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    parsed = urlparse(
        value if "://" in value else "https://" + value
    )

    host = (parsed.hostname or "").lower().rstrip(".")

    if not host:
        return None

    path = parsed.path.rstrip("/")

    return f"{host}{path}"


def get_domain(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    parsed = urlparse(
        value if "://" in value else "https://" + value
    )

    return (parsed.hostname or "").lower().rstrip(".") or None


def load_column(path, column):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = (
        pd.read_excel(path)
        if path.suffix.lower() == ".xlsx"
        else pd.read_csv(path)
    )

    if column not in df.columns:
        raise ValueError(
            f"Missing '{column}' column in {path.name}. "
            f"Found: {list(df.columns)}"
        )

    return df[column].dropna().astype(str)


datasets = {
    name: load_column(path, column)
    for name, (path, column) in FILES.items()
}

comp_urls = set(
    datasets["CompPhish"].map(normalize_url).dropna()
)

comp_hostnames = set(
    datasets["CompPhish"].map(get_domain).dropna()
)

print("\nCOMP PHISH OVERLAP AUDIT")
print("------------------------")
print("CompPhish unique normalized URLs:", len(comp_urls))
print("CompPhish unique hostnames:", len(comp_hostnames))

for name in ["Mendeley", "Original"]:

    other_urls = set(
        datasets[name].map(normalize_url).dropna()
    )

    other_hostnames = set(
        datasets[name].map(get_domain).dropna()
    )

    print(f"\nCompPhish vs {name}")
    print("------------------------")
    print(
        "Normalized URL overlaps:",
        len(comp_urls & other_urls)
    )
    print(
        "Hostname overlaps:",
        len(comp_hostnames & other_hostnames)
    )