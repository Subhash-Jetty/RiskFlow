# Model Card

## Objective
The Loan Performance Intelligence Engine models predict key loan events including delinquency (3M/6M), default (12M), prepayment (12M), and next-state transitions, alongside anomaly exception scoring.

## Data & Features
- **Data Input**: The system utilizes monthly performance panels linked with static origination attributes and external servicer updates. 
- **Features**: Raw inputs include balance metrics, credit/LTV/DTI bands, state, loan purpose, and property type.
- **Engineered Features**: Key derived features include rolling days-past-due (DPD) averages, balance change ratios, target-encoded categoricals, and servicer conflict flags.

## Model Types
- **Supervised Predictions**: Extreme Gradient Boosting (`XGBClassifier`) handles the primary multi-outcome predictions due to its non-linear modeling capacity and resilience to missing values. Logistic Regression is utilized as a robust baseline.
- **Unsupervised Anomaly**: Isolation Forest combined with deterministic rule parsing from `validation_rules.json`.
- **Survival**: Cox Proportional Hazards model for time-to-default risk analysis.

## Validation Method
- **Time-Aware Splitting**: The dataset is chronologically split (70% Train, 15% Val, 15% Test) sorting strictly by `reporting_month` to mimic out-of-time production inference.

## Metrics
Models are evaluated across a diverse suite of metrics designed for class-imbalanced financial tabular data:
- `ROC-AUC` and `PR-AUC`
- `F1 Score` and `Recall at Precision (0.90)`
- `Brier Score` (Binary Calibration)
- `Macro-F1` (Multiclass Prediction)

## Known Limitations
- The synthetic dataset fallback produces highly stylized relationships that may not fully encapsulate complex macroeconomic interactions.
- Isolation Forests tend to struggle with high-dimensional sparse representations; hence feature selection is crucial.

## Leakage Controls
- Target labels (e.g., `next_3m_delinquency_flag`) are strictly dropped before training.
- Rolling features (e.g., rolling DPD) are calculated using strictly historical windows via `.expanding()` groupings to avoid looking ahead.
- Target Encoding smoothing is exclusively fitted on the training split.

## Failure Modes
- Unprecedented macroeconomic shocks (e.g., sudden interest rate spikes) may shift the distribution outside the trained domain, causing degraded calibration. The pipeline mitigates this partially via scenario simulations, but it does not auto-retrain dynamically.
