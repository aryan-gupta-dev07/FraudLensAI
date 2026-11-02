import pytest

from src.url_features import extract_url_features


def test_extract_features_from_valid_url():
    features = extract_url_features("https://example.com")

    assert features["url_length"] == 19
    assert features["hostname_length"] == 11
    assert features["path_length"] == 0
    assert features["subdomain_count"] == 0
    assert features["https_scheme"] == 1


def test_extract_features_from_url_without_scheme():
    features = extract_url_features("example.com")

    assert features["https_scheme"] == 1
    assert features["hostname_length"] == 11


def test_detect_ip_address():
    features = extract_url_features("http://192.168.1.1")

    assert features["uses_ip_address"] == 1


def test_detect_subdomains():
    features = extract_url_features(
        "https://accounts.example.com/login"
    )

    assert features["subdomain_count"] == 1
    assert features["path_length"] == 6


def test_detect_at_symbol():
    features = extract_url_features(
        "https://user@example.com"
    )

    assert features["contains_at_symbol"] == 1


def test_detect_dash_in_hostname():
    features = extract_url_features(
        "https://example-site.com"
    )

    assert features["contains_dash_in_hostname"] == 1


@pytest.mark.parametrize("url", ["", "   "])
def test_reject_empty_url(url):
    with pytest.raises(ValueError):
        extract_url_features(url)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "not a valid url",
    ],
)
def test_reject_invalid_url(url):
    with pytest.raises(ValueError):
        extract_url_features(url)