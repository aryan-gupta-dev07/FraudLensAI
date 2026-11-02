FraudLens AI — Final Evaluation Summary

1. Project Overview

FraudLens AI is a machine-learning-based phishing URL detection system. It extracts URL structure features and estimates whether a URL may be suspicious. The application does not visit or open submitted websites.

2. Final CompPhish Evaluation

Decision threshold: 0.40

Metric| Result
Evaluated URLs| 13,267
True Positives| 5,024
True Negatives| 6,321
False Positives| 998
False Negatives| 924
Accuracy| 85.51%
Precision| 83.43%
Recall| 84.47%
F1-score| 83.94%

3. Error Analysis

The evaluation produced 998 false positives and 924 false negatives. Error-analysis files were saved for further inspection.

These errors show that URL structure alone cannot reliably identify every phishing or legitimate URL.

4. Previous Evaluation

A separate domain-separated evaluation reported 97.55% accuracy. This result must not be combined with the CompPhish result because the evaluation datasets and protocols differ.

5. Limitations

- Performance depends on the dataset and its distribution.
- New phishing patterns may not be detected.
- Legitimate URLs may receive elevated risk scores.
- The risk score is a model output, not a verified real-world probability.
- The system does not verify website content or guarantee that a URL is safe.
- Further independent, time-based and real-world validation is required before production use.

6. Conclusion

FraudLens AI demonstrates a complete URL-analysis workflow, feature extraction, machine-learning inference, threshold-based classification, automated testing and cross-dataset evaluation.

The CompPhish evaluation achieved 85.51% accuracy, 83.43% precision, 84.47% recall and 83.94% F1-score at a threshold of 0.40.

This is a research and portfolio project, not a production-grade security guarantee.