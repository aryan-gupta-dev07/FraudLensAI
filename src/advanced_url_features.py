import ipaddress
import re
from urllib.parse import urlparse

import tldextract


SUSPICIOUS_TOKENS = {
    "login", "signin", "verify", "verification",
    "secure", "account", "update", "confirm",
    "password", "banking", "credential", "auth",
    "payment", "wallet", "webscr", "recover",
}

BRAND_TOKENS = {
    "paypal", "google", "microsoft", "apple",
    "amazon", "facebook", "instagram", "netflix",
    "linkedin", "americanexpress", "skype",
}

extractor = tldextract.TLDExtract(suffix_list_urls=())


def extract_advanced_url_features(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL cannot be empty.")

    url = url.strip()

    candidate = (
        url if url.lower().startswith(("http://", "https://"))
        else "https://" + url
    )

    candidate = re.sub(r"\s", "%20", candidate)

    parsed = urlparse(candidate)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Invalid HTTP or HTTPS URL.")

    hostname = parsed.hostname.lower()

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    host_without_www = hostname.removeprefix("www.")
    path_query = (parsed.path + " " + parsed.query).lower()

    host_tokens = set(re.findall(r"[a-z0-9]+", host_without_www))
    path_tokens = set(re.findall(r"[a-z0-9]+", path_query))

    extracted = extractor(hostname)
    registered_domain = extracted.domain.lower()

    embedded_domains = re.findall(
        r"(?:[a-z0-9-]+\.)+[a-z]{2,}",
        path_query,
    )

    embedded_brand_count = sum(
        brand in path_tokens for brand in BRAND_TOKENS
    )

    hostname_brand_count = sum(
        brand in host_tokens for brand in BRAND_TOKENS
    )

    suspicious_token_count = sum(
        token in path_tokens for token in SUSPICIOUS_TOKENS
    )

    features = {
        "url_length": len(candidate),
        "hostname_length": len(hostname),
        "path_length": len(parsed.path),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "uses_ip_address": is_ip,
        "contains_at_symbol": int("@" in candidate),
        "contains_dash_in_hostname": int("-" in hostname),
        "https_scheme": int(parsed.scheme == "https"),
        "digit_count": sum(char.isdigit() for char in candidate),
        "special_character_count": len(
            re.findall(r"[^a-zA-Z0-9]", candidate)
        ),
        "hostname_token_count": len(host_tokens),
        "path_token_count": len(path_tokens),
        "suspicious_token_count": suspicious_token_count,
        "embedded_domain_count": len(embedded_domains),
        "embedded_brand_count": embedded_brand_count,
        "hostname_brand_count": hostname_brand_count,
        "brand_mismatch": int(
            embedded_brand_count > 0 and hostname_brand_count == 0
        ),
        "hostname_has_digit": int(any(c.isdigit() for c in hostname)),
        "registered_domain_length": len(registered_domain),
    }

    return features