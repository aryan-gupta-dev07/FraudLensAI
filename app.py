import streamlit as st
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from predict_url import predict_url


st.set_page_config(
    page_title="FraudLens AI",
    page_icon="🔍",
    layout="centered"
)

st.title("FraudLens AI")
st.subheader("AI-Powered URL Risk Analysis")

st.write(
    "Enter a URL to analyze its structural features "
    "and receive a machine-learning prediction."
)

st.warning(
    "This tool analyzes URL structure only. "
    "It does not visit websites. Predictions are not "
    "a guarantee that a URL is safe or malicious."
)

url = st.text_input(
    "Enter a URL",
    placeholder="https://example.com"
)

if st.button("Analyze URL"):

    if not url.strip():
        st.error("Please enter a URL.")

    else:
        try:
            result = predict_url(url)

            st.divider()
            st.header("Analysis Results")

            st.write(
                "Prediction:",
                result["prediction"]
            )

            st.metric(
                "Model Risk Score",
                f'{result["risk_score"]:.2f}%'
            )

            st.caption(
                "This score is the model's estimate for its "
                "suspicious class, not a verified real-world "
                "phishing probability."
            )

            st.subheader("Extracted URL Features")

            features = [
                {"Feature": name, "Value": value}
                for name, value in result["features"].items()
            ]

            st.dataframe(
                features,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "No website was visited during this analysis."
            )

        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"An unexpected error occurred: {error}"
            )

st.divider()
st.caption("FraudLens AI | URL Structure-Based Analysis")