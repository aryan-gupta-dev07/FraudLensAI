from urllib.parse import urlparse
import ipaddress
import re


def extract_url_features(url):
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    candidate = url if "://" in url else "https://" + url

    if any(char.isspace() for char in candidate):
        raise ValueError(
            "Please enter a valid HTTP or HTTPS URL."
        )

    parsed = urlparse(candidate)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please enter a valid HTTP or HTTPS URL.")

    hostname = parsed.hostname.lower()

    try:
        ipaddress.ip_address(hostname)
        uses_ip_address = True
    except ValueError:
        uses_ip_address = False

    features = {
        "url_length": len(candidate),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "uses_ip_address": int(uses_ip_address),
        "contains_at_symbol": int("@" in candidate),
        "contains_dash_in_hostname": int("-" in hostname),
        "https_scheme": int(parsed.scheme == "https"),
        "digit_count": sum(char.isdigit() for char in candidate),
        "special_character_count": len(
            re.findall(r"[^a-zA-Z0-9]", candidate)
        ),
    }

    return features


if __name__ == "__main__":
    test_urls = [
        "https://example.com",
        "https://accounts.example.com/login",
    ]

    for test_url in test_urls:
        print("\nURL:", test_url)

        features = extract_url_features(test_url)

        for name, value in features.items():
            print(f"{name}: {value}")