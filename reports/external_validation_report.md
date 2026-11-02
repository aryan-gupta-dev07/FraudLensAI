# FraudLens AI — External Validation Report

## 1. Objective

The objective of this experiment was to evaluate the cross-dataset
generalization capability of the existing FraudLens URL structural
classification model.

The model was trained using the original PhishStorm-derived dataset.
It was evaluated on a separate URL-Phish dataset without retraining.

No URLs were visited during this experiment.

## 2. Dataset Information

### Original Dataset

- Source: PhishStorm
- Cleaned records: 95,900
- Features: 10 structural URL features
- Labels: 0 = Benign, 1 = Phishing

### External Dataset

- Source: URL-Phish, Mendeley Data
- Total records: 116,600
- Benign records: 100,000
- Phishing records: 16,600
- Columns: 26
- Missing values: 14, all in the TLD column
- Duplicate rows: 1,369

## 3. URL Overlap Analysis

An exact URL comparison was performed between the two datasets.

- Unique external URLs: 115,227
- Unique original URLs: 95,890
- Exact overlapping URLs: 0
- Exact URL overlap percentage: 0%

This establishes zero exact URL overlap after basic
lowercase and whitespace normalization.

It does not establish complete dataset independence or
exclude shared domains, sources, or collection biases.

## 4. Evaluation Methodology

The existing FraudLens URL structural model was loaded
without retraining.

The model pipeline contains:

- SimpleImputer
- RandomForestClassifier

Ten structural features were extracted locally from the
external dataset URLs using the existing feature extractor.

All 116,600 records were successfully processed.

No external websites were accessed.

## 5. External Evaluation Results

- Evaluation records: 116,600
- Accuracy: 28.56%

- Benign precision: 89.46%
- Benign recall: 18.93%
- Benign F1-score: 31.25%

- Phishing precision: 15.06%
- Phishing recall: 86.57%
- Phishing F1-score: 25.65%

- Macro F1-score: 28.45%
- Weighted F1-score: 30.45%

### Confusion Matrix

| Actual / Predicted | Benign | Phishing |
|---|---:|---:|
| Benign | 18,928 | 81,072 |
| Phishing | 2,230 | 14,370 |

The model generated a high number of false-positive
predictions on the external dataset.

## 6. Prediction Distribution

| Category | Actual | Predicted |
|---|---:|---:|
| Benign | 100,000 | 21,158 |
| Phishing | 16,600 | 95,442 |

The predicted class distribution differs substantially
from the actual class distribution.

## 7. Feature Distribution Analysis

The following values represent overall feature means calculated across all records in each dataset, without separating benign and phishing classes.

Selected feature means:

| Feature | Original | External |
|---|---:|---:|
| URL length | 72.452 | 32.953 |
| Hostname length | 22.348 | 17.405 |
| Path length | 25.836 | 6.391 |
| HTTPS scheme | 1.000 | 0.431 |
| Digit count | 8.169 | 0.944 |
| Special characters | 11.710 | 7.002 |

These differences indicate substantial differences
in feature distributions between the datasets.

## 8. Error Analysis

### False Positives

- Count: 81,072
- Mean URL length: 26.961
- Mean path length: 2.347
- Mean digit count: 0.098

### False Negatives

- Count: 2,230
- Mean URL length: 42.744
- Mean path length: 16.153
- Mean digit count: 3.070

The error analysis indicates that the model frequently
classified benign external URLs as phishing.

## 9. Key Findings

1. The existing model showed substantially lower
   performance on the external dataset.

2. Exact URL overlap between the two datasets was zero.

3. Feature distributions differed considerably.

4. The model generated a large number of false positives.

5. Same-source evaluation results did not guarantee
   cross-dataset generalization.

## 10. Limitations

- Zero exact URL overlap does not prove complete
  dataset independence.
- Both datasets may contain shared collection biases.
- Feature means do not fully describe distributions.
- The experiment evaluates one external dataset.
- Results should not be interpreted as real-world
  production performance.
- The model has not yet been calibrated or optimized
  for the external dataset.
- Domain analysis identified 90,860 unique domain values,
  including 24 domains containing both benign and phishing
  labels. The external dataset's domain field also includes
  shared hosting and platform domains, which may cause
  conservative grouping during domain-wise evaluation.  

## 11. Conclusion

The external evaluation revealed a substantial
generalization gap between the original evaluation
environment and the external dataset.

The findings highlight the importance of external
validation, dataset diversity, and error analysis
when developing URL-based phishing detection systems.

The original trained model was preserved without
retraining or modification during this experiment.