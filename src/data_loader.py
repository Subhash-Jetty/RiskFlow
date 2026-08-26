import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

from src import config

def check_data_exists() -> bool:
    """Check if required data files exist."""
    required_files = [
        config.TRAIN_FILE,
        config.TEST_FILE,
        config.STATIC_FILE,
        config.SERVICER_FILE,
        config.VALIDATION_RULES_FILE,
        config.MACRO_SCENARIOS_FILE,
        config.SUBMISSION_TEMPLATE_FILE,
        config.DATA_DICTIONARY_FILE
    ]
    return all(f.exists() for f in required_files)


def generate_synthetic_data() -> None:
    """Generate all required data files in data/ directory with schema-matching synthetic data."""
    print("Generating synthetic data...")
    rng = np.random.RandomState(config.RANDOM_SEED)

    num_loans = config.SYNTH_NUM_LOANS
    max_months = config.SYNTH_NUM_MONTHS

    # Generate loan IDs
    loan_ids = [f"L{str(i).zfill(6)}" for i in range(1, num_loans + 1)]

    # Generate static attributes for each loan
    # Origination spanning 2018-01 to 2023-12 (72 months)
    origination_months = []
    base_date = datetime(2018, 1, 1)
    for _ in range(num_loans):
        month_offset = rng.randint(0, 72)
        orig_date = base_date + pd.DateOffset(months=month_offset)
        origination_months.append(orig_date.strftime("%Y-%m"))

    # Sort loans by origination month to easily split last 20% for test
    sorted_indices = np.argsort(origination_months)
    loan_ids = [loan_ids[i] for i in sorted_indices]
    origination_months = [origination_months[i] for i in sorted_indices]

    num_test = int(num_loans * 0.2)
    train_loans = set(loan_ids[:-num_test])
    test_loans = set(loan_ids[-num_test:])

    static_rows = []
    monthly_rows_train = []
    monthly_rows_test = []

    for i, loan_id in enumerate(loan_ids):
        orig_month = origination_months[i]
        orig_date = datetime.strptime(orig_month, "%Y-%m")

        # Static attributes
        orig_balance = rng.randint(150, 501) * 1000
        credit_band = rng.choice(config.CREDIT_BANDS)
        ltv_band = rng.choice(config.LTV_BANDS)
        dti_band = rng.choice(config.DTI_BANDS)
        state = rng.choice(config.STATES)
        purpose = rng.choice(config.LOAN_PURPOSES)
        occupancy = rng.choice(config.OCCUPANCY_TYPES)
        prop_type = rng.choice(config.PROPERTY_TYPES)
        orig_term = 360
        orig_rate = round(rng.uniform(2.5, 7.0), 2)

        static_rows.append({
            config.COL_LOAN_ID: loan_id,
            config.COL_ORIG_BALANCE: orig_balance,
            config.COL_CREDIT_BAND: credit_band,
            config.COL_LTV_BAND: ltv_band,
            config.COL_DTI_BAND: dti_band,
            config.COL_STATE: state,
            config.COL_LOAN_PURPOSE: purpose,
            config.COL_OCCUPANCY: occupancy,
            config.COL_PROPERTY_TYPE: prop_type,
            config.COL_ORIGINATION_MONTH: orig_month,
            "original_term_months": orig_term,
            "original_interest_rate": orig_rate
        })

        # Base default probability proxy for this loan based on static
        risk_score = 0
        if credit_band in ["<620", "620-659"]: risk_score += 2
        if ltv_band in [">95", "91-95"]: risk_score += 1

        # Simulate monthly records
        curr_balance = orig_balance
        status_seq = ["Current", "30DPD", "60DPD", "90DPD", "Default"]
        curr_status_idx = 0

        num_months_active = rng.randint(10, max_months + 1)
        for m in range(num_months_active):
            rep_date = orig_date + pd.DateOffset(months=m)
            reporting_month = rep_date.strftime("%Y-%m")
            loan_age = m + 1
            remaining_term = max(orig_term - loan_age, 0)
            
            # Simple amortization approx
            if curr_balance > 0:
                curr_balance -= orig_balance / orig_term
                curr_balance = max(curr_balance, 0)

            # Evolve status
            dpd = 0
            if curr_status_idx > 0 and curr_status_idx < 4:
                dpd = curr_status_idx * 30
            elif curr_status_idx == 4:
                dpd = 120

            current_status = status_seq[curr_status_idx]

            # Features
            mod_flag = 1 if rng.rand() < 0.05 else 0
            prepay_flag = 1 if current_status == "Prepaid" else 0
            default_flag = 1 if current_status == "Default" else 0
            
            loss_band = "None"
            if default_flag:
                loss_band = rng.choice(config.LOSS_BANDS[1:])

            row = {
                config.COL_LOAN_ID: loan_id,
                config.COL_MONTH_INDEX: m,
                config.COL_REPORTING_MONTH: reporting_month,
                config.COL_ORIGINATION_MONTH: orig_month,
                config.COL_LOAN_AGE: loan_age,
                config.COL_REMAINING_TERM: remaining_term,
                config.COL_ORIG_BALANCE: orig_balance,
                config.COL_CURR_BALANCE: curr_balance,
                config.COL_INTEREST_RATE: orig_rate,
                config.COL_CREDIT_BAND: credit_band,
                config.COL_LTV_BAND: ltv_band,
                config.COL_DTI_BAND: dti_band,
                config.COL_STATE: state,
                config.COL_LOAN_PURPOSE: purpose,
                config.COL_OCCUPANCY: occupancy,
                config.COL_PROPERTY_TYPE: prop_type,
                config.COL_SERVICER: rng.choice(config.SERVICER_NAMES),
                config.COL_STATUS: current_status,
                config.COL_DPD: dpd,
                config.COL_MOD_FLAG: mod_flag,
                config.COL_PREPAY_FLAG: prepay_flag,
                config.COL_DEFAULT_FLAG: default_flag,
                config.COL_LOSS_BAND: loss_band,
                config.COL_LAST_UPDATED: datetime.now().isoformat(),
                config.COL_SOURCE_SYSTEM: rng.choice(config.SOURCE_SYSTEMS),
                config.COL_DOC_STATUS: rng.choice(config.DOC_STATUSES)
            }

            if loan_id in train_loans:
                # Add targets correlated with state
                next_3m_del = 1 if curr_status_idx > 0 or rng.rand() < 0.1 * (risk_score + 1) else 0
                next_6m_del = 1 if curr_status_idx > 0 or rng.rand() < 0.15 * (risk_score + 1) else 0
                next_12m_def = 1 if curr_status_idx > 1 or rng.rand() < 0.05 * (risk_score + 1) else 0
                next_12m_prepay = 1 if curr_status_idx == 0 and rng.rand() < 0.1 else 0
                
                next_state_idx = curr_status_idx
                if current_status not in ["Default", "Prepaid"]:
                    if rng.rand() < 0.1 + (risk_score * 0.05):
                        next_state_idx = min(curr_status_idx + 1, 4)
                    elif rng.rand() < 0.05:
                        next_state_idx = 0 # cure
                        if rng.rand() < 0.1: # prepay
                            current_status = "Prepaid"
                            
                next_st = status_seq[next_state_idx] if current_status != "Prepaid" else "Prepaid"
                
                exception_req = 1 if rng.rand() < 0.05 else 0
                exception_type = "None"
                if exception_req:
                    exception_type = rng.choice(config.EXCEPTION_TYPES[1:])

                row[config.TARGET_NEXT_3M_DEL] = next_3m_del
                row[config.TARGET_NEXT_6M_DEL] = next_6m_del
                row[config.TARGET_NEXT_12M_DEF] = next_12m_def
                row[config.TARGET_NEXT_12M_PREPAY] = next_12m_prepay
                row[config.TARGET_NEXT_STATE] = next_st
                row[config.TARGET_EXCEPTION_REQ] = exception_req
                row[config.TARGET_EXCEPTION_TYPE] = exception_type
                
                monthly_rows_train.append(row)
            else:
                monthly_rows_test.append(row)
                
            # Transition for next month
            if current_status not in ["Default", "Prepaid"]:
                if rng.rand() < 0.1 + (risk_score * 0.05):
                    curr_status_idx = min(curr_status_idx + 1, 4)
                elif rng.rand() < 0.05:
                    curr_status_idx = 0
                    if rng.rand() < 0.05:
                        curr_status_idx = 5 # Hack to signify prepay for next iteration logic
                        status_seq.append("Prepaid")

    df_train = pd.DataFrame(monthly_rows_train)
    df_test = pd.DataFrame(monthly_rows_test)
    df_static = pd.DataFrame(static_rows)

    # Insert ~5% missing values in non-ID columns for train
    for col in df_train.columns:
        if col not in [config.COL_LOAN_ID, config.COL_REPORTING_MONTH, config.COL_ORIGINATION_MONTH] and col not in config.ALL_TARGETS:
            mask = rng.rand(len(df_train)) < 0.05
            if df_train[col].dtype == object:
                df_train.loc[mask, col] = np.nan
            else:
                df_train.loc[mask, col] = np.nan

    for col in df_test.columns:
        if col not in [config.COL_LOAN_ID, config.COL_REPORTING_MONTH, config.COL_ORIGINATION_MONTH]:
            mask = rng.rand(len(df_test)) < 0.05
            if df_test[col].dtype == object:
                df_test.loc[mask, col] = np.nan
            else:
                df_test.loc[mask, col] = np.nan

    df_train.to_csv(config.TRAIN_FILE, index=False)
    df_test.to_csv(config.TEST_FILE, index=False)
    df_static.to_csv(config.STATIC_FILE, index=False)

    # Servicer Updates
    servicer_rows = []
    for _ in range(2000):
        lid = rng.choice(loan_ids)
        servicer_rows.append({
            config.COL_LOAN_ID: lid,
            "update_date": (datetime.now() - timedelta(days=rng.randint(1, 100))).strftime("%Y-%m-%d"),
            "field_updated": rng.choice(["current_balance", "interest_rate", "current_status"]),
            "old_value": rng.randint(100000, 200000),
            "new_value": rng.randint(90000, 190000),
            config.COL_SERVICER: rng.choice(config.SERVICER_NAMES),
            config.COL_SOURCE_SYSTEM: rng.choice(config.SOURCE_SYSTEMS)
        })
    pd.DataFrame(servicer_rows).to_csv(config.SERVICER_FILE, index=False)

    # Validation Rules
    rules = [
        {"rule_id": "R001", "field": "current_balance", "condition": "current_balance <= original_balance * 1.05", "severity": "high", "description": "Current balance should not exceed original balance by more than 5%"},
        {"rule_id": "R002", "field": "loan_age_months", "condition": "loan_age_months >= 0", "severity": "high", "description": "Loan age cannot be negative"},
        {"rule_id": "R003", "field": "days_past_due", "condition": "days_past_due >= 0", "severity": "high", "description": "DPD cannot be negative"},
        {"rule_id": "R004", "field": "interest_rate", "condition": "interest_rate > 0 and interest_rate < 20", "severity": "medium", "description": "Interest rate should be between 0 and 20"},
        {"rule_id": "R005", "field": "original_balance", "condition": "original_balance > 0", "severity": "high", "description": "Original balance must be positive"},
        {"rule_id": "R006", "field": "current_status", "condition": "current_status in ['Current', '30DPD', '60DPD', '90DPD', 'Default', 'Prepaid']", "severity": "high", "description": "Invalid current status"},
        {"rule_id": "R007", "field": "credit_score_band", "condition": "credit_score_band in ['<620', '620-659', '660-699', '700-739', '740-779', '780+']", "severity": "medium", "description": "Invalid credit score band"},
        {"rule_id": "R008", "field": "ltv_band", "condition": "ltv_band in ['<=60', '61-70', '71-80', '81-90', '91-95', '>95']", "severity": "medium", "description": "Invalid LTV band"},
        {"rule_id": "R009", "field": "dti_band", "condition": "dti_band in ['<=20', '21-30', '31-40', '41-50', '>50']", "severity": "medium", "description": "Invalid DTI band"},
        {"rule_id": "R010", "field": "document_status", "condition": "document_status in ['Complete', 'Partial', 'Missing', 'Under Review']", "severity": "low", "description": "Invalid document status"}
    ]
    with open(config.VALIDATION_RULES_FILE, "w") as f:
        json.dump(rules, f, indent=4)

    # Macro Scenarios
    macro_data = {
        "scenario_name": ["base", "adverse_credit", "high_prepayment"],
        "interest_rate_shift": [0.0, 2.0, -1.5],
        "unemployment_shift": [0.0, 3.0, 0.0],
        "hpi_shift": [0.0, -10.0, 5.0],
        "credit_tightening": [0, 1, 0]
    }
    pd.DataFrame(macro_data).to_csv(config.MACRO_SCENARIOS_FILE, index=False)

    # Submission Template
    pd.DataFrame(columns=config.SUBMISSION_COLUMNS).to_csv(config.SUBMISSION_TEMPLATE_FILE, index=False)

    # Data Dictionary — full field definitions for RAG grounding
    dd_content = """# Data Dictionary

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
"""
    with open(config.DATA_DICTIONARY_FILE, "w") as f:
        f.write(dd_content)

    print("Synthetic data generation complete.")


def load_all_data() -> dict:
    """
    Load all data files, merge static attributes onto monthly panel.
    Returns dict with keys: 'train', 'test', 'static', 'servicer', 
    'validation_rules', 'macro_scenarios', 'submission_template', 'data_dictionary'
    If data/ is empty, calls generate_synthetic_data() first.
    """
    if not check_data_exists():
        generate_synthetic_data()

    print("Loading datasets...")
    static_df = pd.read_csv(config.STATIC_FILE)
    
    train_df = pd.read_csv(config.TRAIN_FILE)
    train_df = pd.merge(train_df, static_df, on=config.COL_LOAN_ID, how='left', suffixes=('', '_static'))
    
    test_df = pd.read_csv(config.TEST_FILE)
    test_df = pd.merge(test_df, static_df, on=config.COL_LOAN_ID, how='left', suffixes=('', '_static'))
    
    servicer_df = pd.read_csv(config.SERVICER_FILE)
    macro_df = pd.read_csv(config.MACRO_SCENARIOS_FILE)
    sub_df = pd.read_csv(config.SUBMISSION_TEMPLATE_FILE)

    with open(config.VALIDATION_RULES_FILE, "r") as f:
        val_rules = json.load(f)

    with open(config.DATA_DICTIONARY_FILE, "r") as f:
        data_dict = f.read()

    return {
        'train': train_df,
        'test': test_df,
        'static': static_df,
        'servicer': servicer_df,
        'validation_rules': val_rules,
        'macro_scenarios': macro_df,
        'submission_template': sub_df,
        'data_dictionary': data_dict
    }
