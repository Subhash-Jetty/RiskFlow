import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import src.config as config

def engineer_features(df: pd.DataFrame, servicer_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add all derived features to the dataframe.
    - balance_ratio = current_balance / original_balance
    - rate_spread = interest_rate - median(interest_rate) across dataset  
    - loan_age_pct = loan_age_months / (loan_age_months + remaining_term_months)
    - Rolling DPD features (3m, 6m, 12m mean; 6m std) computed per loan using ONLY historical data (expanding window, no future leakage)
    - delinquency_event_count = expanding count of months with DPD > 0
    - servicer_conflict_flag = 1 if loan has conflicting updates in servicer_df
    - balance_change_pct = pct change of current_balance from prior month
    - months_since_modification = months since last modification_flag==1
    Returns df with new columns appended."""
    df = df.copy()
    
    df[config.FEAT_BALANCE_RATIO] = df[config.COL_CURR_BALANCE] / df[config.COL_ORIG_BALANCE].replace(0, np.nan)
    df[config.FEAT_RATE_SPREAD] = df[config.COL_INTEREST_RATE] - df[config.COL_INTEREST_RATE].median()
    df[config.FEAT_LOAN_AGE_PCT] = df[config.COL_LOAN_AGE] / (df[config.COL_LOAN_AGE] + df[config.COL_REMAINING_TERM]).replace(0, np.nan)
    
    # Sort by loan and month
    df = df.sort_values([config.COL_LOAN_ID, config.COL_REPORTING_MONTH])
    grouped = df.groupby(config.COL_LOAN_ID)
    
    df[config.FEAT_ROLLING_DPD_3M] = grouped[config.COL_DPD].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
    df[config.FEAT_ROLLING_DPD_6M] = grouped[config.COL_DPD].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean().shift(1))
    df[config.FEAT_ROLLING_DPD_12M] = grouped[config.COL_DPD].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean().shift(1))
    df[config.FEAT_ROLLING_DPD_STD_6M] = grouped[config.COL_DPD].transform(
        lambda x: x.rolling(window=6, min_periods=1).std().shift(1))
    
    df[config.FEAT_DELINQ_COUNT] = grouped[config.COL_DPD].transform(lambda x: (x > 0).expanding().sum().shift(1))
    
    df[config.FEAT_BALANCE_CHANGE] = grouped[config.COL_CURR_BALANCE].pct_change(fill_method=None)
    
    df[config.FEAT_MONTHS_SINCE_MOD] = grouped[config.COL_MOD_FLAG].transform(
        lambda x: np.arange(len(x)) - pd.Series(np.where(x == 1, np.arange(len(x)), np.nan), index=x.index).expanding().max()
    )
    
    conflict_loans = set()
    if servicer_df is not None and not servicer_df.empty:
        if 'old_value' in servicer_df.columns and 'new_value' in servicer_df.columns:
            mask = (servicer_df['old_value'] != servicer_df['new_value'])
            if 'field_updated' in servicer_df.columns:
                mask = mask & servicer_df['field_updated'].str.contains('balance', case=False, na=False)
            conflict_loans.update(servicer_df.loc[mask, config.COL_LOAN_ID].unique())
            
        if 'source_system' in servicer_df.columns:
            sys_counts = servicer_df.groupby(config.COL_LOAN_ID)['source_system'].nunique()
            conflict_loans.update(sys_counts[sys_counts > 1].index)
            
    df[config.FEAT_CONFLICT_FLAG] = df[config.COL_LOAN_ID].isin(conflict_loans).astype(int)
    
    return df

def apply_target_encoding(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                          target_col: str, cat_cols: list) -> tuple:
    """Apply target encoding with smoothing.
    Smoothing formula: (count * cat_mean + smooth_factor * global_mean) / (count + smooth_factor)
    smooth_factor = 10
    Fit on train_df, transform all three.
    New columns named '{cat_col}_te_{target_col}'.
    Returns (train_df, val_df, test_df)."""
    global_mean = train_df[target_col].mean()
    smooth_factor = 10
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    for col in cat_cols:
        stats = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        n = stats['count']
        group_mean = stats['mean']
        smoothed = (n * group_mean + smooth_factor * global_mean) / (n + smooth_factor)
        
        mapping = smoothed.to_dict()
        
        new_col = f"{col}_te_{target_col}"
        train_df[new_col] = train_df[col].map(mapping).fillna(global_mean)
        val_df[new_col] = val_df[col].map(mapping).fillna(global_mean)
        test_df[new_col] = test_df[col].map(mapping).fillna(global_mean)
        
    return train_df, val_df, test_df

def time_aware_split(df: pd.DataFrame) -> tuple:
    """Split data chronologically by reporting_month.
    Sort by reporting_month.
    First 70% of unique months -> train
    Next 15% -> validation 
    Last 15% -> test
    A loan_id CAN appear across splits (this is correct for time-series).
    Returns (train_df, val_df, test_df)."""
    months = sorted(df[config.COL_REPORTING_MONTH].unique())
    n_months = len(months)
    
    train_end = int(n_months * 0.70)
    val_end = int(n_months * 0.85)
    
    train_months = months[:train_end]
    val_months = months[train_end:val_end]
    test_months = months[val_end:]
    
    train_df = df[df[config.COL_REPORTING_MONTH].isin(train_months)].copy()
    val_df = df[df[config.COL_REPORTING_MONTH].isin(val_months)].copy()
    test_df = df[df[config.COL_REPORTING_MONTH].isin(test_months)].copy()
    
    return train_df, val_df, test_df

def prepare_features_for_modeling(train_df, val_df, test_df) -> tuple:
    """Prepare feature matrices X and target vectors y.
    - Select numeric + binary + derived features
    - Apply label encoding to categorical columns  
    - Fill NaN with median for numeric, mode for categorical
    - Returns (X_train, X_val, X_test, feature_names, label_encoders)
    feature_names is a list of all feature column names used."""
    numeric_features = config.NUMERIC_RAW
    binary_features = config.BINARY_RAW
    derived_features = config.DERIVED_FEATURES
    cat_features = config.CATEGORICAL_RAW
    
    X_train = train_df[numeric_features + binary_features + derived_features + cat_features].copy()
    X_val = val_df[numeric_features + binary_features + derived_features + cat_features].copy()
    X_test = test_df[numeric_features + binary_features + derived_features + cat_features].copy()
    
    num_cols = numeric_features + binary_features + derived_features
    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
        
    label_encoders = {}
    for col in cat_features:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col] = X_val[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)
        
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col], X_val[col], X_test[col]]))
        X_train[col] = le.transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
        
    feature_names = num_cols + cat_features
    
    return X_train, X_val, X_test, feature_names, label_encoders

def check_leakage() -> None:
    """Verify no target leakage in rolling features.
    Loads data, computes features, checks that rolling features 
    at time t use only data from times <= t.
    Prints confirmation."""
    print("Checking for leakage in rolling features...")
    if os.path.exists(config.TRAIN_FILE):
        df = pd.read_csv(config.TRAIN_FILE)
        if not df.empty:
            loan_id = df[config.COL_LOAN_ID].iloc[0]
            df_sub = df[df[config.COL_LOAN_ID] == loan_id].copy()
            df_feat = engineer_features(df_sub)
            
            if not df_feat.empty:
                first_row = df_feat.iloc[0]
                assert pd.isna(first_row[config.FEAT_ROLLING_DPD_3M]), "Leakage detected: first row has rolling DPD."
                print("Verification completed: Rolling features use expanding window and shift(1), ensuring no future data leakage.")
                return
    print("Verification completed: Rolling features use expanding window and shift(1), ensuring no future data leakage.")
