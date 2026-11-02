from pathlib import Path
import pandas as pd

from predict_url import load_model, load_threshold
from advanced_url_features import extract_advanced_url_features

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "independent" / "tranco_clean.csv"

SAMPLE_SIZE = 5000

df = pd.read_csv(DATA_PATH)

# Use a reproducible sample instead of processing one million domains.
sample = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
).copy()

model = load_model()
threshold = load_threshold()

feature_names = list(model.feature_names_in_)

print("Domains selected:", len(sample))
print("Model threshold:", threshold)

feature_rows = []

for domain in sample["domain"]:
    url = "https://" + str(domain)

    features = extract_advanced_url_features(url)

    feature_rows.append([
        features[name] for name in feature_names
    ])

X = pd.DataFrame(feature_rows, columns=feature_names)

probabilities = model.predict_proba(X)

classes = list(model.classes_)
suspicious_index = classes.index(1)

scores = probabilities[:, suspicious_index]

predictions = scores >= threshold

# Save flagged domains for manual review.
flagged = sample.loc[predictions, ["rank", "domain"]].copy()

flagged["risk_score"] = (
    scores[predictions] * 100
).round(2)

report_path = BASE_DIR / "reports" / "tranco_flagged_review.csv"

flagged.to_csv(report_path, index=False)

print("\nFlagged-domain review saved to:")
print(report_path)

print("\nTop 10 flagged domains:")
print(
    flagged.sort_values(
        "risk_score",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nTRanco screening results")
print("------------------------")
print("Total domains:", len(sample))
print("Flagged suspicious:", int(predictions.sum()))
print("Classified likely legitimate:", int((~predictions).sum()))

print(
    "Flagged percentage:",
    round(float(predictions.mean() * 100), 2),
    "%"
)

print("\nNote: Tranco domains are not verified safety labels.")
print("These results are not accuracy or false-positive-rate metrics.")