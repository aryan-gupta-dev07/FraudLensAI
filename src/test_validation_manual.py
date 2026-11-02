from predict_url import predict_url

test_urls = [
    "",
    "not a valid url",
    "https://",
    "ftp://example.com",
    "https://example.com"
]

for url in test_urls:
    try:
        result = predict_url(url)
        print(repr(url), "-> ACCEPTED:", result["prediction"])

    except ValueError as error:
        print(repr(url), "-> REJECTED:", error)