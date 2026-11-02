from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "urlset_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "fraudlens_model.pkl"

df = pd.read_csv(DATA_PATH)

# Use numeric URL features; exclude domain and label.
X = df.drop(columns=["domain", "label"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))
y = df["label"].astype(int)

# Split before training to evaluate on unseen records.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print("Training model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== MODEL EVALUATION ===")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)
print("Training complete.")