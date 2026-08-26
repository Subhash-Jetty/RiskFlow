# Data Dictionary

## Loan Identifiers
- **loan_id**: Unique identifier for each loan in the portfolio. Format: L followed by 6 digits.
- **month_index**: Sequential integer identifying the reporting period (0-based).
- **reporting_month**: Calendar month of the observation in YYYY-MM format.

## Origination Attributes
- **origination_month**: The month the loan was originated, in YYYY-MM format.
- **original_balance**: The original unpaid principal balance at origination, in USD. Typical range: $100,000 – $600,000.
- **original_term_months**: The original contractual term of the loan in months (e.g., 360 for a 30-year mortgage).
- **original_interest_rate**: The note rate at origination, expressed as a percentage.

## Borrower & Property
- **credit_score_band**: Borrower credit score grouped into bands: <620, 620-659, 660-699, 700-739, 740-779, 780+. Lower bands indicate higher credit risk.
- **ltv_band**: Loan-to-value ratio band at origination: <=60, 61-70, 71-80, 81-90, 91-95, >95. Higher LTV implies less borrower equity and higher risk.
- **dti_band**: Debt-to-income ratio band: <=20, 21-30, 31-40, 41-50, >50. Higher DTI indicates greater borrower leverage.
- **state**: US state where the property is located (2-letter abbreviation).
- **loan_purpose**: Purpose of the loan: Purchase, Refinance, or Cash-Out Refinance.
- **occupancy_type**: Property occupancy: Primary, Second Home, or Investment.
- **property_type**: Type of property: Single Family, Condo, 2-4 Unit, or Manufactured.

## Monthly Performance
- **loan_age_months**: Number of months since origination.
- **remaining_term_months**: Remaining contractual term in months.
- **current_balance**: Current unpaid principal balance. Should generally decrease over time due to amortisation.
- **interest_rate**: Current note rate (may differ from original if modified).
- **days_past_due**: Number of days the borrower is past due on payments. 0 = current, 30/60/90/120+ = delinquent.
- **current_status**: Loan performance status: Current, 30DPD, 60DPD, 90DPD, Default, or Prepaid. Represents the most recent delinquency bucket.
- **modification_flag**: Binary (0/1). 1 if the loan has been modified (e.g., rate reduction, term extension).
- **prepayment_flag**: Binary (0/1). 1 if the loan was prepaid (borrower paid off early).
- **default_flag**: Binary (0/1). 1 if the loan has defaulted (typically 180+ DPD or foreclosure).
- **loss_severity_band**: Estimated loss given default: None, 0-20%, 20-40%, 40-60%, 60-80%, 80-100%.

## Servicing & Documentation
- **servicer_name**: Name of the loan servicer (Servicer_A through Servicer_D).
- **last_updated_at**: Timestamp of the last data update for this record.
- **source_system**: The originating data system (SystemA or SystemB). Used for conflict detection.
- **document_status**: Completeness of loan documentation: Complete, Partial, Missing, or Under Review.

## Target Variables
- **next_3m_delinquency_flag**: Binary. 1 if the loan becomes 60+ DPD within the next 3 months.
- **next_6m_delinquency_flag**: Binary. 1 if the loan becomes 60+ DPD within the next 6 months.
- **next_12m_default_flag**: Binary. 1 if the loan defaults within the next 12 months.
- **next_12m_prepayment_flag**: Binary. 1 if the loan prepays within the next 12 months.
- **next_state**: The loan's status in the next reporting period: Current, 30DPD, 60DPD, 90DPD, Default, Prepaid.
- **exception_required**: Binary. 1 if the record requires human review due to data quality or anomaly concerns.
- **exception_type**: Category of exception: None, Balance_Discrepancy, Status_Conflict, Document_Gap, Payment_Anomaly.
