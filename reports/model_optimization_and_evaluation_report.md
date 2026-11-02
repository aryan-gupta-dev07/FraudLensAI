Model Optimization and Evaluation Report

1. Objective

This study evaluates alternative machine learning classifiers, investigates feature importance, examines additional engineered features, and optimizes the existing URL structural classification model.

The primary objective is to improve classification performance while preserving the original trained model and maintaining reproducible evaluation procedures.

2. Dataset Overview

The experiments used the cleaned URL structural feature dataset.

- Total records: 95,900
- Original features: 10
- Target variable: "label"
- Training and testing split: 80:20
- Random state: 42
- Stratification: Enabled

The dataset contains two classes: benign (0) and phishing (1).

3. Model Comparison

Four machine learning classifiers were evaluated using 3-fold Stratified Cross-Validation.

The evaluated classifiers were:

1. Random Forest
2. Extra Trees
3. Logistic Regression
4. Hist Gradient Boosting

3.1 Cross-Validation Results

Model| Accuracy| Precision| Recall| F1-score
Random Forest| 88.39%| 90.16%| 86.15%| 88.11%
Hist Gradient Boosting| 88.37%| 91.87%| 84.16%| 87.85%
Extra Trees| 88.02%| 89.08%| 86.64%| 87.84%
Logistic Regression| 79.05%| 84.34%| 71.27%| 77.26%

Random Forest achieved the highest mean F1-score, while Hist Gradient Boosting achieved the highest precision.

4. Feature Importance Analysis

Feature importance was extracted from the existing Random Forest classifier.

Feature| Importance
URL length| 27.09%
Special character count| 16.24%
Path length| 15.53%
Subdomain count| 14.85%
Hostname length| 12.52%
Digit count| 12.37%
Dash in hostname| 1.34%
At symbol| 0.06%
IP address usage| 0.00%
HTTPS scheme| 0.00%

The results indicate that URL length and other structural characteristics contributed most strongly to the fitted model's predictions.

Zero importance does not establish that a feature is universally irrelevant; it indicates that the fitted model did not use that feature for its tree splits.

5. Feature Engineering Experiment

Five additional features were derived from the original structural features:

- Hostname-to-URL length ratio
- Path-to-URL length ratio
- Digit-to-URL length ratio
- Special-character-to-URL length ratio
- Path presence indicator

The original and engineered feature sets were evaluated using the same Random Forest configuration and 3-fold Stratified Cross-Validation.

5.1 Results

Metric| Original Features| Engineered Features
Feature count| 10| 15
Accuracy| 88.39%| 88.31%
Precision| 90.16%| 90.08%
Recall| 86.15%| 86.07%
F1-score| 88.11%| 88.03%

The engineered feature set did not improve classification performance.

Consequently, the original 10-feature representation was retained for subsequent optimization experiments.

6. Hyperparameter Optimization

Grid Search Cross-Validation was performed on the Random Forest classifier.

The following parameters were investigated:

Parameter| Tested Values
Number of estimators| 100, 200
Maximum depth| None, 20
Minimum samples split| 2, 5
Minimum samples leaf| 1, 2

A total of 16 parameter configurations were evaluated using 3-fold Stratified Cross-Validation, resulting in 48 model fits.

6.1 Selected Configuration

Parameter| Selected Value
n_estimators| 200
max_depth| None
min_samples_split| 2
min_samples_leaf| 2
class_weight| balanced
random_state| 42

Best mean cross-validation F1-score: 88.52%

The baseline cross-validation F1-score was 88.11%.

This represents an improvement of approximately 0.41 percentage points.

7. Baseline and Optimized Model Evaluation

Both models were evaluated using the same stratified 80:20 train-test split, with random state 42.

The test set contained 19,180 records.

7.1 Performance Comparison

Metric| Baseline Model| Optimized Model
Accuracy| 88.50%| 88.92%
Phishing Precision| 90.60%| 91.51%
Phishing Recall| 85.91%| 85.78%
Phishing F1-score| 88.20%| 88.55%
Training time| Existing model| 5.09 seconds

The optimized model achieved a modest improvement in accuracy, precision, and phishing F1-score.

However, phishing recall decreased slightly.

7.2 Confusion Matrix Comparison

Baseline Model

| Predicted Benign| Predicted Phishing
Actual Benign| 8,746| 856
Actual Phishing| 1,349| 8,229

Optimized Model

| Predicted Benign| Predicted Phishing
Actual Benign| 8,839| 763
Actual Phishing| 1,362| 8,216

7.3 Error Analysis

The optimized model produced:

- 93 fewer false positives.
- 13 additional false negatives.

This indicates a modest reduction in false alarms, accompanied by a slight increase in missed phishing classifications.

Neither metric should be considered independently when evaluating a phishing detection system.

8. Experimental Reproducibility

The experiments used:

- Fixed random seeds.
- Stratified data splitting.
- Stratified cross-validation.
- Consistent evaluation metrics.
- Separate experimental scripts.
- CSV and text-based result reports.

The original trained model was preserved throughout the experiments.

The optimized model was evaluated in memory and was not saved over the existing production-facing model file.

9. Limitations

1. The experiments primarily evaluate performance on the existing dataset.
2. Cross-validation and holdout results measure different evaluation procedures and should not be directly interchanged.
3. The feature importance values describe the fitted model and do not establish causal relationships.
4. The engineered features did not demonstrate measurable improvement.
5. Hyperparameter optimization produced a modest improvement rather than a substantial performance increase.
6. Further independent and domain-aware validation is necessary to assess generalization.
7. Dataset-specific accuracy does not establish real-world phishing detection reliability.

10. Conclusion

The experiments established a reproducible comparison of four machine learning classifiers and investigated the contribution of URL structural features.

Random Forest achieved the strongest cross-validation F1-score among the evaluated models.

Feature engineering did not improve performance, while hyperparameter optimization produced a modest improvement in holdout accuracy and phishing F1-score.

The optimized configuration reduced false positives but introduced a small increase in false negatives.

The original model remained preserved, and the experimental results were documented separately.

Further work should prioritize independent validation, generalization analysis, and prediction reliability before selecting an optimized model for practical deployment.