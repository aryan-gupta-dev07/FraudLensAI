import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.url_features import extract_url_features

df = pd.read_csv(BASE_DIR / "data" / "urlset_clean.csv")

success = 0
failed = []

for index, url in enumerate(df["domain"].dropna()):
    url = str(url).strip()

    if any(char.isspace() for char in url):
        continue

    candidate = (
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
    )

    try:
        features = extract_url_features(candidate)
        if len(features) == 10:
            success += 1
        else:
            failed.append((index, "Incorrect feature count"))
    except Exception as error:
        failed.append((index, str(error)))

print("Successful feature extraction:", success)
print("Failed:", len(failed))
print("Expected features per URL: 10")

if failed:
    print("\nFirst failures:")
    for item in failed[:10]:
        print(item)