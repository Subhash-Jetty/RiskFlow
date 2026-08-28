import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from src import config

def prepare_competing_risk_data(df):
    """Prepare dataframe for competing risk analysis."""
    last_obs = df.groupby(config.COL_LOAN_ID).last().reset_index()
    last_obs['duration'] = last_obs[config.COL_LOAN_AGE]
    
    conditions = [
        last_obs[config.COL_STATUS] == 'Default',
        last_obs[config.COL_STATUS] == 'Prepaid'
    ]
    choices = [1, 2]
    last_obs['event_type'] = np.select(conditions, choices, default=0)
    
    cols_to_keep = [
        config.COL_LOAN_ID, 'duration', 'event_type',
        config.COL_INTEREST_RATE, config.COL_ORIG_BALANCE,
        config.COL_CREDIT_BAND, config.COL_LTV_BAND, config.COL_DTI_BAND
    ]
    return last_obs[cols_to_keep]

def fit_cause_specific_hazards(cr_df):
    """Fit cause-specific Cox models for default and prepayment."""
    df_encoded = cr_df.copy()
    df_encoded[config.COL_CREDIT_BAND] = pd.Categorical(df_encoded[config.COL_CREDIT_BAND], categories=config.CREDIT_BANDS, ordered=True).codes
    df_encoded[config.COL_LTV_BAND] = pd.Categorical(df_encoded[config.COL_LTV_BAND], categories=config.LTV_BANDS, ordered=True).codes
    df_encoded[config.COL_DTI_BAND] = pd.Categorical(df_encoded[config.COL_DTI_BAND], categories=config.DTI_BANDS, ordered=True).codes
    
    df_encoded = df_encoded.drop(columns=[config.COL_LOAN_ID])
    
    df_default = df_encoded.copy()
    df_default['event'] = (df_default['event_type'] == 1).astype(int)
    df_default = df_default.drop(columns=['event_type'])
    
    df_prepay = df_encoded.copy()
    df_prepay['event'] = (df_prepay['event_type'] == 2).astype(int)
    df_prepay = df_prepay.drop(columns=['event_type'])
    
    cox_default = CoxPHFitter(penalizer=0.1).fit(df_default, duration_col='duration', event_col='event')
    cox_prepay = CoxPHFitter(penalizer=0.1).fit(df_prepay, duration_col='duration', event_col='event')
    
    return {
        'cox_default': cox_default,
        'cox_prepay': cox_prepay,
        'concordance_default': cox_default.concordance_index_,
        'concordance_prepay': cox_prepay.concordance_index_,
        'summary_default': cox_default.summary,
        'summary_prepay': cox_prepay.summary
    }

def compute_cumulative_incidence(cox_default, cox_prepay, cr_df):
    """Compute cumulative incidence functions from cause-specific hazards."""
    bh_default = cox_default.baseline_hazard_
    bh_prepay = cox_prepay.baseline_hazard_
    
    times = np.unique(np.concatenate([bh_default.index.values, bh_prepay.index.values]))
    
    bh_def = bh_default.reindex(times, fill_value=0.0).values.flatten()
    bh_prep = bh_prepay.reindex(times, fill_value=0.0).values.flatten()
    
    ch_def = np.cumsum(bh_def)
    ch_prep = np.cumsum(bh_prep)
    
    s_overall = np.exp(-(ch_def + ch_prep))
    s_prev = np.insert(s_overall[:-1], 0, 1.0)
    
    cif_default = np.cumsum(s_prev * bh_def)
    cif_prepay = np.cumsum(s_prev * bh_prep)
    
    return {
        'times': times,
        'cif_default': cif_default,
        'cif_prepay': cif_prepay
    }

def plot_competing_risks(cif_results):
    """Plot Cumulative Incidence Functions."""
    plt.figure(figsize=(10, 6))
    plt.plot(cif_results['times'], cif_results['cif_default'], label='Default')
    plt.plot(cif_results['times'], cif_results['cif_prepay'], label='Prepayment')
    plt.xlabel('Time (Months)')
    plt.ylabel('Cumulative Incidence')
    plt.title('Competing Risks: Default vs Prepayment')
    plt.legend()
    plt.grid(True)
    
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.REPORT_DIR / 'competing_risk_cif.png')
    plt.close()

def run_competing_risk_analysis(df):
    """Run full competing risk analysis pipeline."""
    cr_df = prepare_competing_risk_data(df)
    models = fit_cause_specific_hazards(cr_df)
    cif_results = compute_cumulative_incidence(models['cox_default'], models['cox_prepay'], cr_df)
    plot_competing_risks(cif_results)
    
    return {
        'cr_df': cr_df,
        'models': models,
        'cif_results': cif_results
    }
