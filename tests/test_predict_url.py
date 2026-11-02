import pytest

from src.predict_url import validate_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "example.com",
        "https://accounts.example.com/login",
    ],
)
def test_accept_valid_urls(url):
    valid, message = validate_url(url)

    assert valid is True
    assert message == "URL format is valid."


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a valid url",
        "ftp://example.com",
        "https://example",
        "https://-example.com",
        "https://example..com",
    ],
)
def test_reject_invalid_urls(url):
    valid, message = validate_url(url)

    assert valid is False
    assert isinstance(message, str)
    assert len(message) > 0


def test_reject_non_string_input():
    valid, message = validate_url(None)

    assert valid is False
    assert message == "URL cannot be empty."


def test_reject_invalid_port():
    valid, message = validate_url("https://example.com:99999")

    assert valid is False
    assert message == "URL contains an invalid port."