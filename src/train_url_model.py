from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "url_features_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraudlens_url_model.pkl"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y,
)

model = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("classifier", RandomForestClassifier(
n_estimators=100,
random_state=42,
class_weight="balanced",
n_jobs=-1,
)),
])

print("Training URL structural model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== URL MODEL EVALUATION ===")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Feature names:", list(X.columns))