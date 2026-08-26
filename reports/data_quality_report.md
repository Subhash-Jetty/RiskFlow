# Data Quality Report

*(auto-generated — all numbers computed at runtime)*

## Column Overview

- **Train columns**: 55
- **Train rows**: 21,980
- **Test rows**: 5,240

## Missingness

| Column | Train Miss % | Test Miss % |
|--------|-------------|------------|
| loan_id | 0.00% | 0.00% |
| month_index | 5.01% | 4.73% |
| reporting_month | 0.00% | 0.00% |
| origination_month | 0.00% | 0.00% |
| loan_age_months | 4.90% | 4.96% |
| remaining_term_months | 4.94% | 4.39% |
| original_balance | 4.84% | 4.47% |
| current_balance | 4.87% | 4.68% |
| interest_rate | 4.98% | 4.92% |
| credit_score_band | 4.92% | 4.85% |
| ltv_band | 4.95% | 4.90% |
| dti_band | 4.98% | 5.31% |
| state | 5.26% | 5.40% |
| loan_purpose | 5.33% | 4.79% |
| occupancy_type | 5.00% | 5.48% |
| property_type | 4.79% | 4.83% |
| servicer_name | 4.87% | 4.77% |
| current_status | 4.89% | 4.75% |
| days_past_due | 4.95% | 5.13% |
| modification_flag | 4.79% | 4.90% |
| prepayment_flag | 5.08% | 5.15% |
| default_flag | 5.15% | 4.73% |
| loss_severity_band | 59.16% | 61.03% |
| last_updated_at | 4.90% | 5.15% |
| source_system | 5.08% | 4.71% |
| document_status | 5.07% | 4.85% |
| next_3m_delinquency_flag | 0.00% | nan% |
| next_6m_delinquency_flag | 0.00% | nan% |
| next_12m_default_flag | 0.00% | nan% |
| next_12m_prepayment_flag | 0.00% | nan% |
| next_state | 0.00% | nan% |
| exception_required | 0.00% | nan% |
| exception_type | 94.92% | nan% |
| original_balance_static | 0.00% | 0.00% |
| credit_score_band_static | 0.00% | 0.00% |
| ltv_band_static | 0.00% | 0.00% |
| dti_band_static | 0.00% | 0.00% |
| state_static | 0.00% | 0.00% |
| loan_purpose_static | 0.00% | 0.00% |
| occupancy_type_static | 0.00% | 0.00% |
| property_type_static | 0.00% | 0.00% |
| origination_month_static | 0.00% | 0.00% |
| original_term_months | 0.00% | 0.00% |
| original_interest_rate | 0.00% | 0.00% |
| balance_ratio | 9.48% | 8.89% |
| rate_spread | 4.98% | 4.92% |
| loan_age_pct | 9.56% | 9.18% |
| rolling_dpd_mean_3m | 1.96% | 2.10% |
| rolling_dpd_mean_6m | 1.93% | 2.08% |
| rolling_dpd_mean_12m | 1.93% | 2.08% |
| rolling_dpd_std_6m | 3.81% | 4.08% |
| delinquency_event_count | 1.82% | 1.91% |
| balance_change_pct | 11.13% | 10.84% |
| months_since_modification | 29.90% | 28.74% |
| servicer_conflict_flag | 0.00% | 0.00% |

## Numeric Distributions

| Column | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| loan_age_months | 34.22 | 23.51 | 1.00 | 100.00 |
| remaining_term_months | 325.83 | 23.51 | 260.00 | 359.00 |
| original_balance | 325769.03 | 104241.98 | 150000.00 | 499000.00 |
| current_balance | 294891.77 | 96915.50 | 112805.56 | 497613.89 |
| interest_rate | 4.75 | 1.31 | 2.50 | 7.00 |
| days_past_due | 68.87 | 50.73 | 0.00 | 120.00 |
| month_index | 33.07 | 23.49 | 0.00 | 99.00 |

## Outliers (IQR Method)

| Column | Lower Outliers | Upper Outliers |
|--------|---------------|---------------|
| loan_age_months | 0 | 0 |
| remaining_term_months | 0 | 0 |
| original_balance | 0 | 0 |
| current_balance | 0 | 0 |
| interest_rate | 0 | 0 |
| days_past_due | 0 | 0 |
| month_index | 0 | 0 |

## Validation Rule Violations

- **R001**: Current balance should not exceed original balance by more than 5%
- **R002**: Loan age cannot be negative
- **R003**: DPD cannot be negative
- **R004**: Interest rate should be between 0 and 20
- **R005**: Original balance must be positive
- **R006**: Invalid current status
- **R007**: Invalid credit score band
- **R008**: Invalid LTV band
- **R009**: Invalid DTI band
- **R010**: Invalid document status

## Train vs Test Drift (Mean Shift)

| Column | Train Mean | Test Mean | Abs Δ |
|--------|-----------|----------|-------|
| loan_age_months | 34.2177 | 32.6293 | 1.5883 |
| remaining_term_months | 325.8320 | 327.3681 | 1.5361 |
| original_balance | 325769.0285 | 322756.4922 | 3012.5363 |
| current_balance | 294891.7743 | 293340.4999 | 1551.2743 |
| interest_rate | 4.7464 | 4.8639 | 0.1175 |
| days_past_due | 68.8742 | 66.3005 | 2.5737 |
| month_index | 33.0700 | 31.6839 | 1.3861 |