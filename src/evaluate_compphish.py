from pathlib import Path
from urllib.parse import urlparse
import json

import joblib
import pandas as pd

from advanced_url_features import extract_advanced_url_features


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "independent" / "comp_phish"
    / "All_Features_threshold90.xlsx"
)

MENDELEY_PATH = BASE_DIR / "data" / "independent" / "Dataset.csv"
ORIGINAL_PATH = BASE_DIR / "data" / "urlset_clean.csv"

MODEL_PATH = BASE_DIR / "models" / "fraudlens_domain_adapted_model.pkl"

THRESHOLD_PATH = (
    BASE_DIR / "models" / "fraudlens_domain_adapted_threshold.json"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def get_hostname(url):
    try:
        url = str(url).strip()

        if not url or url.lower() == "nan":
            return None

        if "://" not in url:
            url = "https://" + url

        hostname = urlparse(url).hostname

        return hostname.lower().rstrip(".") if hostname else None

    except Exception:
        return None


def normalize_url(url):
    try:
        url = str(url).strip()

        if not url or url.lower() == "nan":
            return None

        if "://" not in url:
            url = "https://" + url

        parsed = urlparse(url)

        if not parsed.hostname:
            return None

        hostname = parsed.hostname.lower().rstrip(".")
        path = parsed.path.rstrip("/")

        return f"{hostname}{path}?{parsed.query}"

    except Exception:
        return None


def load_hostnames(file_path):

    data = pd.read_csv(file_path)

    possible_columns = {
        "url",
        "domain",
        "hostname",
        "domain_name"
    }

    url_column = next(
        (
            column for column in data.columns
            if str(column).strip().lower() in possible_columns
        ),
        None
    )

    if url_column is None:
        raise ValueError(
            f"No URL/domain column found in {file_path}. "
            f"Available columns: {list(data.columns)}"
        )

    print(
        f"Using column '{url_column}' "
        f"from {file_path.name}"
    )

    hostnames = data[url_column].map(get_hostname)

    return set(hostnames.dropna())


def main():

    print("Loading CompPhish dataset...")

    data = pd.read_excel(DATA_PATH)

    required_columns = {"url", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Missing required columns: "
            f"{required_columns - set(data.columns)}"
        )

    print("Original rows:", len(data))

    if data["label"].isna().any():
        raise ValueError("Dataset contains missing labels.")

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected label values found.")

    print("\nLoading existing dataset hostnames...")

    mendeley_hosts = load_hostnames(MENDELEY_PATH)
    original_hosts = load_hostnames(ORIGINAL_PATH)

    excluded_hosts = mendeley_hosts | original_hosts

    print("Mendeley unique hostnames:", len(mendeley_hosts))
    print("Original unique hostnames:", len(original_hosts))
    print("Combined excluded hostnames:", len(excluded_hosts))

    print("\nRemoving overlapping hostnames...")

    data["hostname"] = data["url"].map(get_hostname)
    data["normalized_url"] = data["url"].map(normalize_url)

    before_overlap = len(data)

    data = data[
        data["hostname"].notna()
        & ~data["hostname"].isin(excluded_hosts)
    ].copy()

    removed_overlap = before_overlap - len(data)

    print("Rows removed due to hostname overlap:", removed_overlap)

    before_duplicates = len(data)

    data = data.dropna(subset=["normalized_url"])

    data = data.drop_duplicates(
        subset=["normalized_url"],
        keep="first"
    ).copy()

    removed_duplicates = before_duplicates - len(data)

    print("Duplicate URLs removed:", removed_duplicates)
    print("Remaining rows:", len(data))

    print("\nRemaining label distribution:")
    print(data["label"].value_counts())

    if data.empty:
        raise RuntimeError("No data remains after overlap removal.")

    if data["label"].nunique() != 2:
        raise RuntimeError(
            "Both legitimate and phishing classes are required."
        )

    print("\nLoading existing FraudLensAI model...")

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    expected_features = list(model.feature_names_in_)

    print("Existing threshold:", threshold)
    print("Expected model features:", len(expected_features))

    results = []
    invalid_urls = 0
    invalid_examples = []

    print("\nExtracting FraudLensAI features...")

    for index, row in data.iterrows():

        try:
            url = str(row["url"]).strip()

            features = extract_advanced_url_features(url)

            results.append({
                "row_index": index,
                "url": url,
                "hostname": row["hostname"],
                "actual_label": int(row["label"]),
                **features
            })

        except Exception as error:

            invalid_urls += 1

            if len(invalid_examples) < 10:
                invalid_examples.append({
                    "row_index": int(index),
                    "error": str(error)
                })

    if not results:
        raise RuntimeError(
            "No valid URLs available for evaluation."
        )

    evaluation_data = pd.DataFrame(results)

    missing_features = (
        set(expected_features) - set(evaluation_data.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing model features: {sorted(missing_features)}"
        )

    X = evaluation_data[expected_features]

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            "Extracted features contain missing or non-numeric values."
        )

    y = evaluation_data["actual_label"]

    if y.nunique() != 2:
        raise ValueError(
            "Evaluation requires both classes."
        )

    print("\nValid URLs:", len(evaluation_data))
    print("Invalid URLs:", invalid_urls)

    print("\nRunning predictions with existing model...")

    probabilities = model.predict_proba(X)[
        :, list(model.classes_).index(1)
    ]

    predictions = (probabilities >= threshold).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        classification_report,
    )

    report = {
        "dataset": "CompPhish",
        "evaluation_type": (
            "Hostname-overlap-excluded evaluation"
        ),
        "label_mapping": {
            "0": "Legitimate",
            "1": "Phishing"
        },
        "original_rows": int(before_overlap),
        "mendeley_unique_hostnames": len(mendeley_hosts),
        "original_unique_hostnames": len(original_hosts),
        "excluded_hostname_count": len(excluded_hosts),
        "overlap_rows_removed": int(removed_overlap),
        "duplicate_urls_removed": int(removed_duplicates),
        "valid_urls": int(len(evaluation_data)),
        "invalid_urls": int(invalid_urls),
        "invalid_examples": invalid_examples,
        "remaining_label_distribution": {
            str(key): int(value)
            for key, value in y.value_counts().items()
        },
        "threshold": threshold,
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            zero_division=0
        ),
    }

    evaluation_data["predicted_label"] = predictions
    evaluation_data["risk_score"] = probabilities

    evaluation_data.to_csv(
        REPORT_DIR / "compphish_predictions.csv",
        index=False
    )

    with open(
        REPORT_DIR / "compphish_overlap_evaluation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nFINAL COMPPHISH OVERLAP EVALUATION")
    print("----------------------------------")
    print(json.dumps(report, indent=4))

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()