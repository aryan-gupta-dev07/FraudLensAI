FraudLens AI

Machine Learning-Based Phishing URL Detection System

FraudLens AI is a machine learning project designed to identify potentially suspicious and phishing URLs using URL-based structural features.

The project focuses on detecting suspicious URL patterns without directly visiting or opening the submitted websites.

Features

- Machine learning-based phishing URL classification.
- URL structural feature extraction.
- Suspicious URL analysis.
- Prediction through an interactive Streamlit interface.
- Model evaluation and performance reports.
- Testing and error analysis.

Technology Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Machine Learning

Model Evaluation

The following results were obtained during the unseen CompPhish dataset evaluation at a classification threshold of 0.40.

Metric| Result
Accuracy| 85.51%
Precision| 83.43%
Recall| 84.47%
F1-score| 83.94%

These results represent a specific dataset evaluation and should not be interpreted as guaranteed real-world performance.

Project Structure

FraudLensAI/
├── app.py
├── requirements.txt
├── src/
├── tests/
├── reports/
└── .gitignore

Installation

Clone the repository:

git clone https://github.com/aryan-gupta-dev07/FraudLensAI.git
cd FraudLensAI

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Model Files and Datasets

Large trained model files and datasets are excluded from this repository to keep the GitHub repository lightweight.

The application requires the corresponding trained model files to be available in the expected local model directory. Refer to the project source code for the required filenames and paths.

Limitations

- Predictions are based primarily on URL structural characteristics.
- The system does not inspect the actual webpage content.
- Performance can vary across datasets and changing phishing techniques.
- Prediction scores are not verified real-world probabilities.
- The system should not be treated as a standalone security guarantee.

Disclaimer

FraudLens AI is an educational and research-oriented project. It is intended for machine learning experimentation and phishing URL analysis, not as a replacement for professional cybersecurity solutions.

Author

Developed as an AI/ML portfolio project.
