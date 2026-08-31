# Stress Sensitivity by Feature Cluster

| Cluster Features | Direction | Default Rate Delta | Delinquency Rate Delta |
|---|---|---|---|
| days_past_due, prepayment_flag, default_flag, rolling_dpd_mean_3m, rolling_dpd_mean_6m, rolling_dpd_mean_12m, delinquency_event_count | down | -0.0868 | -0.0976 |
| current_status | down | -0.0225 | -0.0202 |
| days_past_due, prepayment_flag, default_flag, rolling_dpd_mean_3m, rolling_dpd_mean_6m, rolling_dpd_mean_12m, delinquency_event_count | up | 0.0202 | 0.0111 |
| original_balance, current_balance | down | 0.0002 | -0.0009 |
| ltv_band | up | 0.0008 | 0.0005 |
| credit_score_band | down | 0.0008 | -0.0001 |
| document_status | down | 0.0004 | -0.0004 |
| months_since_modification | up | 0.0004 | 0.0003 |
| loan_age_months, loan_age_pct | down | -0.0004 | -0.0000 |
| months_since_modification | down | -0.0001 | -0.0004 |
| remaining_term_months | up | -0.0003 | -0.0000 |
| original_balance, current_balance | up | -0.0002 | 0.0003 |
| dti_band | down | -0.0001 | -0.0003 |
| document_status | up | -0.0003 | 0.0003 |
| state | up | -0.0000 | -0.0003 |
| interest_rate, rate_spread | down | 0.0001 | 0.0003 |
| servicer_name | down | 0.0003 | 0.0002 |
| dti_band | up | 0.0001 | 0.0002 |
| credit_score_band | up | -0.0001 | -0.0002 |
| month_index | up | 0.0001 | 0.0002 |
| servicer_name | up | -0.0001 | -0.0002 |
| property_type | down | -0.0001 | -0.0002 |
| loan_purpose | down | -0.0002 | 0.0000 |
| month_index | down | -0.0002 | -0.0001 |
| remaining_term_months | down | 0.0002 | 0.0000 |
| loan_age_months, loan_age_pct | up | 0.0001 | -0.0001 |
| occupancy_type | down | 0.0001 | -0.0000 |
| property_type | up | 0.0000 | 0.0001 |
| interest_rate, rate_spread | up | 0.0000 | -0.0001 |
| rolling_dpd_std_6m | up | 0.0001 | -0.0000 |
| ltv_band | down | 0.0000 | -0.0001 |
| servicer_conflict_flag | down | 0.0000 | 0.0001 |
| rolling_dpd_std_6m | down | -0.0000 | 0.0000 |
| balance_ratio, balance_change_pct | down | -0.0000 | 0.0000 |
| state | down | -0.0000 | -0.0000 |
| balance_ratio, balance_change_pct | up | -0.0000 | 0.0000 |
| modification_flag | down | 0.0000 | 0.0000 |
| modification_flag | up | 0.0000 | 0.0000 |
| servicer_conflict_flag | up | 0.0000 | 0.0000 |
| loan_purpose | up | 0.0000 | 0.0000 |
| occupancy_type | up | 0.0000 | 0.0000 |
| current_status | up | 0.0000 | 0.0000 |
| loss_severity_band | up | 0.0000 | 0.0000 |
| loss_severity_band | down | 0.0000 | 0.0000 |