from src.predict_url import (
    predict_url,
    load_model,
    load_threshold,
)

from src.advanced_url_features import (
    extract_advanced_url_features,
)


def test_advanced_extractor_returns_19_features():
    features = extract_advanced_url_features(
        "https://example.com"
    )

    assert len(features) == 19


def test_features_match_model_requirements():
    model = load_model()

    features = extract_advanced_url_features(
        "https://example.com"
    )

    assert set(features.keys()) == set(
        model.feature_names_in_
    )


def test_saved_threshold_is_correct():
    threshold = load_threshold()

    assert abs(threshold - 0.60) < 0.001


def test_domain_adapted_prediction():
    result = predict_url("https://example.com")

    assert result["prediction"] in [
        "Potentially suspicious",
        "Likely legitimate",
    ]

    assert 0 <= result["risk_score"] <= 100
    assert len(result["features"]) == 19
    assert result["threshold"] == load_threshold()