FraudLens AI

Machine Learning-Based Phishing URL Detection System

FraudLens AI is a machine learning project designed to identify potentially suspicious and phishing URLs using URL-based structural features.

The system analyzes URL characteristics to classify potentially malicious links without directly visiting or opening the submitted websites.

Features

- Machine learning-based phishing URL classification.
- URL structural feature extraction.
- Suspicious URL analysis.
- Interactive Streamlit application.
- Model evaluation and performance reports.
- Error analysis and testing.
- External dataset evaluation.

Technology Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Machine Learning

Model Evaluation

The following results were obtained during evaluation on the unseen CompPhish dataset using a classification threshold of 0.40.

Metric| Result
Accuracy| 85.51%
Precision| 83.43%
Recall| 84.47%
F1-score| 83.94%

These results represent a specific dataset evaluation. They should not be interpreted as guaranteed real-world performance.

Project Structure

FraudLensAI/
├── app.py
├── requirements.txt
├── src/
├── tests/
├── reports/
├── models/       # Local trained models (not included)
├── data/         # Local datasets (not included)
└── .gitignore

Installation

1. Clone the repository

git clone https://github.com/aryan-gupta-dev07/FraudLensAI.git
cd FraudLensAI

2. Create a virtual environment

python -m venv .venv

3. Activate the environment

Windows:

.venv\Scripts\activate

Linux/macOS:

source .venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

Running the Application

After installing the required dependencies and placing the trained model files in the expected local directory, run:

streamlit run app.py

Models and Datasets

Large trained model files and datasets are excluded from this GitHub repository to keep it lightweight.

The application requires the corresponding trained model files to be available locally. Refer to the source code for the expected model filenames and paths.

Limitations

- Predictions are primarily based on URL structural characteristics.
- The system does not inspect actual webpage content.
- Performance can vary across datasets and changing phishing techniques.
- Prediction scores are not verified real-world probabilities.
- The system is not a standalone security guarantee.

Disclaimer

FraudLens AI is an educational and research-oriented machine learning project intended for phishing URL analysis and experimentation.

It is not a replacement for professional cybersecurity solutions or browser security protections.
