from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "reports" / "compphish_predictions.csv"

data = pd.read_csv(DATA_PATH)

threshold = 0.40
data["actual_label"] = data["actual_label"].astype(int)
data["predicted_at_040"] = (data["risk_score"] >= threshold).astype(int)

false_negatives = data[
    (data["actual_label"] == 1) &
    (data["predicted_at_040"] == 0)
]

false_positives = data[
    (data["actual_label"] == 0) &
    (data["predicted_at_040"] == 1)
]

print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))

print("\nFalse negative examples:")
print(false_negatives[
    ["url", "risk_score"]
].head(10).to_string(index=False))

print("\nFalse positive examples:")
print(false_positives[
    ["url", "risk_score"]
].head(10).to_string(index=False))

output = BASE_DIR / "reports"
false_negatives.to_csv(output / "compphish_false_negatives_040.csv", index=False)
false_positives.to_csv(output / "compphish_false_positives_040.csv", index=False)

print("\nError analysis files saved.")
