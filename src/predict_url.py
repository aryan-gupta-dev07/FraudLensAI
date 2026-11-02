from pathlib import Path
from urllib.parse import urlparse
from functools import lru_cache
import json
import re

import joblib
import pandas as pd

try:
    from .advanced_url_features import extract_advanced_url_features
except ImportError:
    from advanced_url_features import extract_advanced_url_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "fraudlens_domain_adapted_model.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR / "models" / "fraudlens_domain_adapted_threshold.json"
)


def validate_url(url):

    if not isinstance(url, str) or not url.strip():
        return False, "URL cannot be empty."

    url = url.strip()

    if any(char.isspace() for char in url):
        return False, "URL must not contain spaces."

    candidate = (
        url if "://" in url
        else "https://" + url
    )

    parsed = urlparse(candidate)

    if parsed.scheme not in ("http", "https"):
        return False, "Only HTTP and HTTPS URLs are supported."

    hostname = parsed.hostname

    if not hostname:
        return False, "Please enter a valid domain."

    try:
        parsed.port
    except ValueError:
        return False, "URL contains an invalid port."

    hostname = hostname.lower().rstrip(".")

    if len(hostname) > 253:
        return False, "Domain name is too long."

    labels = hostname.split(".")

    if len(labels) < 2:
        return False, "Please enter a valid domain."

    for label in labels:
        if not label or len(label) > 63:
            return False, "Domain contains an invalid section."

        if not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
            label
        ):
            return False, "Domain contains invalid characters."

    return True, "URL format is valid."


@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold():

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid model threshold.")

    return threshold


def predict_url(url):

    valid, message = validate_url(url)

    if not valid:
        raise ValueError(message)

    features = extract_advanced_url_features(url)

    model = load_model()
    threshold = load_threshold()

    expected_features = list(model.feature_names_in_)

    if set(features.keys()) != set(expected_features):
        raise ValueError(
            "Extracted features do not match model requirements."
        )

    input_data = pd.DataFrame(
        [[features[name] for name in expected_features]],
        columns=expected_features
    )

    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain suspicious class.")

    risk_score = float(probabilities[classes.index(1)])

    prediction = int(risk_score >= threshold)

    return {
        "prediction": (
            "Potentially suspicious"
            if prediction == 1
            else "Likely legitimate"
        ),
        "risk_score": round(risk_score * 100, 2),
        "threshold": threshold,
        "features": features,
    }