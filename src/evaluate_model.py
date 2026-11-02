from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "urlset_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudlens_model.pkl"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["domain", "label"])
X = X.apply(pd.to_numeric, errors="coerce")

model = joblib.load(MODEL_PATH)
classifier = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": classifier.feature_importances_
})

importance = importance.sort_values(
    "Importance", ascending=False
)

print("=== TOP 10 IMPORTANT FEATURES ===")
print(importance.head(10).to_string(index=False))