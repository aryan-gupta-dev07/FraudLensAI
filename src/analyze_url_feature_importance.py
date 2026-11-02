from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraudlens_url_model.pkl"
DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
REPORT_PATH = BASE_DIR / "reports" / "url_feature_importance.csv"

print("Loading original URL model...")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["label"])

classifier = model.named_steps["classifier"]

importance = classifier.feature_importances_

results = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance,
})

results = results.sort_values(
    "Importance",
    ascending=False,
)

print("\n=== URL FEATURE IMPORTANCE ===")
print(results.to_string(index=False))

results.to_csv(REPORT_PATH, index=False)

print("\nResults saved to:", REPORT_PATH)