import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

from src import config

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare survival analysis data from the monthly panel.
    For each loan_id, compute:
    - duration: number of months observed (loan_age_months at last observation)
    - event: 1 if loan defaulted, 0 otherwise (right-censored)
    Also keep static features: credit_score_band, ltv_band, dti_band, state, interest_rate, original_balance
    Returns one row per loan."""
    
    # Sort by loan_id and loan_age_months to get the last observation
    df_sorted = df.sort_values(by=[config.COL_LOAN_ID, config.COL_LOAN_AGE])
    
    # Get the last row for each loan
    last_obs = df_sorted.drop_duplicates(subset=[config.COL_LOAN_ID], keep='last').copy()
    
    # Determine event and censoring
    # Explicitly document censoring treatment
    print("Censoring treatment:")
    print(" - Loans with current_status == 'Default' are events (event=1)")
    print(" - Loans with current_status in ['Current', '30DPD', '60DPD', '90DPD'] are right-censored (event=0)")
    print(" - Loans with current_status == 'Prepaid' are right-censored (event=0)")
    
    last_obs['event'] = np.where(last_obs[config.COL_STATUS] == 'Default', 1, 0)
    last_obs['duration'] = last_obs[config.COL_LOAN_AGE]
    
    keep_cols = [
        config.COL_LOAN_ID, 'duration', 'event', 
        config.COL_CREDIT_BAND, config.COL_LTV_BAND, config.COL_DTI_BAND, 
        config.COL_STATE, config.COL_INTEREST_RATE, config.COL_ORIG_BALANCE
    ]
    
    surv_df = last_obs[keep_cols].copy()
    surv_df = surv_df.dropna(subset=['duration', 'event'])
    surv_df['duration'] = surv_df['duration'].clip(lower=1)  # ensure positive durations
    return surv_df

def fit_cox_model(surv_df: pd.DataFrame) -> tuple:
    """Fit CoxPH model from lifelines.
    Features: interest_rate, original_balance, and label-encoded credit_score_band, ltv_band, dti_band.
    Returns (cox_model, concordance_index, summary_df)."""
    
    df_cox = surv_df.copy()
    
    # Drop NaN rows
    df_cox = df_cox.dropna()
    
    # Encode categoricals
    df_cox[config.COL_CREDIT_BAND] = df_cox[config.COL_CREDIT_BAND].astype('category').cat.set_categories(config.CREDIT_BANDS, ordered=True).cat.codes
    df_cox[config.COL_LTV_BAND] = df_cox[config.COL_LTV_BAND].astype('category').cat.set_categories(config.LTV_BANDS, ordered=True).cat.codes
    df_cox[config.COL_DTI_BAND] = df_cox[config.COL_DTI_BAND].astype('category').cat.set_categories(config.DTI_BANDS, ordered=True).cat.codes
    
    # Handle state if it's there but not mentioned in features list, drop it and loan_id
    features = [
        'duration', 'event', config.COL_INTEREST_RATE, config.COL_ORIG_BALANCE,
        config.COL_CREDIT_BAND, config.COL_LTV_BAND, config.COL_DTI_BAND
    ]
    df_cox = df_cox[features]
    
    # Ensure no -1 codes due to missing categories
    df_cox = df_cox[(df_cox[config.COL_CREDIT_BAND] >= 0) & (df_cox[config.COL_LTV_BAND] >= 0) & (df_cox[config.COL_DTI_BAND] >= 0)]
    
    cph = CoxPHFitter(penalizer=0.01)
    
    try:
        cph.fit(df_cox, duration_col='duration', event_col='event')
        concordance_index = cph.concordance_index_
        summary_df = cph.summary
        print(f"CoxPH Model Concordance Index: {concordance_index:.4f}")
        return cph, concordance_index, summary_df
    except Exception as e:
        print(f"Error fitting CoxPH model: {e}")
        return None, 0.0, pd.DataFrame()

def fit_baseline_logreg(surv_df: pd.DataFrame) -> tuple:
    """Fit logistic regression baseline for 12-month default prediction.
    event_12m = 1 if loan defaulted within 12 months.
    Returns (logreg_model, roc_auc, accuracy)."""
    
    df_lr = surv_df.copy()
    df_lr = df_lr.dropna()
    
    # Create target: defaulted within 12 months
    df_lr['defaulted_within_12m'] = np.where((df_lr['duration'] <= 12) & (df_lr['event'] == 1), 1, 0)
    
    # Encode features
    df_lr[config.COL_CREDIT_BAND] = df_lr[config.COL_CREDIT_BAND].astype('category').cat.set_categories(config.CREDIT_BANDS, ordered=True).cat.codes
    df_lr[config.COL_LTV_BAND] = df_lr[config.COL_LTV_BAND].astype('category').cat.set_categories(config.LTV_BANDS, ordered=True).cat.codes
    df_lr[config.COL_DTI_BAND] = df_lr[config.COL_DTI_BAND].astype('category').cat.set_categories(config.DTI_BANDS, ordered=True).cat.codes
    
    features = [
        config.COL_INTEREST_RATE, config.COL_ORIG_BALANCE,
        config.COL_CREDIT_BAND, config.COL_LTV_BAND, config.COL_DTI_BAND
    ]
    
    X = df_lr[features]
    y = df_lr['defaulted_within_12m']
    
    if y.nunique() <= 1:
        print("Warning: Only one class in target for baseline model.")
        return None, 0.0, 0.0
        
    lr_params = config.LOGREG_PARAMS.copy()
    lr_params.pop('n_jobs', None)
    lr = LogisticRegression(**lr_params)
    try:
        lr.fit(X, y)
        preds = lr.predict(X)
        probs = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        acc = accuracy_score(y, preds)
        print(f"Baseline LogReg ROC-AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        return lr, auc, acc
    except Exception as e:
        print(f"Error fitting Baseline LogReg: {e}")
        return None, 0.0, 0.0

def generate_km_curves(surv_df: pd.DataFrame) -> dict:
    """Generate Kaplan-Meier survival curves segmented by:
    - credit_score_band (all bands)
    - ltv_band (all bands)
    Saves plots to config.REPORT_DIR / 'km_curves_credit.png' and 'km_curves_ltv.png'.
    Returns dict with KM fitted objects."""
    
    km_results = {}
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Credit Score Band
    plt.figure(figsize=(10, 6))
    for band in config.CREDIT_BANDS:
        mask = surv_df[config.COL_CREDIT_BAND] == band
        if not mask.any():
            continue
        kmf = KaplanMeierFitter()
        try:
            seg = surv_df[mask].dropna(subset=['duration', 'event'])
            if len(seg) < 2:
                continue
            kmf.fit(seg['duration'], event_observed=seg['event'], label=band)
            kmf.plot_survival_function()
            km_results[f'credit_{band}'] = kmf
        except Exception as e:
            print(f"Error generating KM curve for Credit Band {band}: {e}")
            
    plt.title('Kaplan-Meier Survival Curves by Credit Score Band')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.REPORT_DIR / 'km_curves_credit.png')
    plt.close()
    
    # LTV Band
    plt.figure(figsize=(10, 6))
    for band in config.LTV_BANDS:
        mask = surv_df[config.COL_LTV_BAND] == band
        if not mask.any():
            continue
        kmf = KaplanMeierFitter()
        try:
            seg = surv_df[mask].dropna(subset=['duration', 'event'])
            if len(seg) < 2:
                continue
            kmf.fit(seg['duration'], event_observed=seg['event'], label=band)
            kmf.plot_survival_function()
            km_results[f'ltv_{band}'] = kmf
        except Exception as e:
            print(f"Error generating KM curve for LTV Band {band}: {e}")
            
    plt.title('Kaplan-Meier Survival Curves by LTV Band')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.REPORT_DIR / 'km_curves_ltv.png')
    plt.close()
    
    return km_results

def run_survival_analysis(df: pd.DataFrame) -> dict:
    """Main entry point. Runs full survival analysis pipeline.
    Returns dict: {'cox_model': ..., 'concordance': ..., 'cox_summary': ..., 
                   'baseline_auc': ..., 'baseline_accuracy': ..., 'km_results': ...}"""
    
    print("Preparing survival data...")
    surv_df = prepare_survival_data(df)
    
    print("\nFitting CoxPH Model...")
    cox_model, concordance, summary = fit_cox_model(surv_df)
    
    print("\nFitting Baseline Logistic Regression...")
    logreg, auc, acc = fit_baseline_logreg(surv_df)
    
    print("\nGenerating Kaplan-Meier Curves...")
    km_results = generate_km_curves(surv_df)
    
    return {
        'cox_model': cox_model,
        'concordance': concordance,
        'cox_summary': summary,
        'baseline_auc': auc,
        'baseline_accuracy': acc,
        'km_results': km_results
    }
