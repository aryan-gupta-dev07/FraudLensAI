from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraudlens_url_model.pkl"

model = joblib.load(MODEL_PATH)

classifier = model.named_steps["classifier"]

feature_names = [
    "url_length",
    "hostname_length",
    "path_length",
    "subdomain_count",
    "uses_ip_address",
    "contains_at_symbol",
    "contains_dash_in_hostname",
    "https_scheme",
    "digit_count",
    "special_character_count",
]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_,
})

importance = importance.sort_values(
    "Importance", ascending=False
)

print("=== URL MODEL FEATURE IMPORTANCE ===")
print(importance.to_string(index=False))

print("\nTotal features:", len(feature_names))