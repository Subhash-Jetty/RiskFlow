import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
try:
    import shap
except ImportError:
    shap = None
import joblib
from pathlib import Path
import json
import logging

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_anomaly_scores(X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Compute Isolation Forest anomaly scores, normalized to [0, 1].
    Uses config.ISOLATION_FOREST_PARAMS.
    Returns array of scores (higher = more anomalous)."""
    # handle NaN in X (fill with median)
    X_filled = X[feature_names].fillna(X[feature_names].median())
    
    if X_filled.empty:
        return np.array([])
        
    model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    model.fit(X_filled)
    
    # Save IF model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_DIR / 'isolation_forest.joblib')
    
    # Scores from decision_function() range from negative (anomaly) to positive (normal).
    # Normalize: anomaly_score = 1 - (score - min) / (max - min), so higher = more anomalous.
    scores = model.decision_function(X_filled)
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score > min_score:
        normalized_scores = 1.0 - ((scores - min_score) / (max_score - min_score))
    else:
        normalized_scores = np.zeros_like(scores)
        
    return normalized_scores

def compute_rule_violations(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """Apply validation_rules.json rules to compute rule violation scores.
    Returns DataFrame with columns: loan_id, month_index, rule_violations_count, 
    rule_violation_details (list of violated rule_ids), rule_score (normalized 0-1)."""
    n_rows = len(df)
    violations_count = np.zeros(n_rows)
    violation_details = [[] for _ in range(n_rows)]
    
    if not rules or df.empty:
        return pd.DataFrame({
            config.COL_LOAN_ID: df.get(config.COL_LOAN_ID, pd.Series(dtype=int)),
            config.COL_MONTH_INDEX: df.get(config.COL_MONTH_INDEX, pd.Series(dtype=int)),
            'rule_violations_count': violations_count,
            'rule_violation_details': violation_details,
            'rule_score': np.zeros(n_rows)
        })
        
    for rule in rules:
        rule_id = rule.get('rule_id')
        condition_str = rule.get('condition', '')
        
        try:
            is_violated = df.eval(condition_str)
            if isinstance(is_violated, pd.Series):
                violation_mask = is_violated.fillna(False).values
                violations_count += violation_mask.astype(int)
                for i in np.where(violation_mask)[0]:
                    violation_details[i].append(rule_id)
        except Exception as e:
            logger.warning(f"Failed to evaluate rule {rule_id} condition '{condition_str}': {e}")
            
    max_possible_violations = len(rules)
    if max_possible_violations > 0:
        rule_score = violations_count / max_possible_violations
    else:
        rule_score = np.zeros(n_rows)
        
    return pd.DataFrame({
        config.COL_LOAN_ID: df[config.COL_LOAN_ID].values,
        config.COL_MONTH_INDEX: df[config.COL_MONTH_INDEX].values,
        'rule_violations_count': violations_count,
        'rule_violation_details': violation_details,
        'rule_score': rule_score
    })

def compute_combined_anomaly_score(if_scores: np.ndarray, rule_scores: np.ndarray) -> np.ndarray:
    """Blend IF and rule scores using config weights.
    combined = config.ANOMALY_IF_WEIGHT * if_scores + config.ANOMALY_RULE_WEIGHT * rule_scores
    Clip to [0, 1]."""
    combined = config.ANOMALY_IF_WEIGHT * if_scores + config.ANOMALY_RULE_WEIGHT * rule_scores
    return np.clip(combined, 0.0, 1.0)

def get_top_anomalies(df: pd.DataFrame, X: pd.DataFrame, combined_scores: np.ndarray, 
                     model=None, feature_names: list = None, n: int = 20) -> pd.DataFrame:
    """Select top-n anomalies and compute their drivers.
    For each anomaly, if a trained model is provided, use SHAP to get top-3 feature contributions.
    Otherwise use IF feature importance.
    Returns DataFrame with: loan_id, month_index, anomaly_score, top_driver_1, top_driver_2, 
    top_driver_3, driver_values, rule_violations."""
    
    if feature_names is None:
        feature_names = X.columns.tolist()
        
    n_actual = min(n, len(df))
    if n_actual == 0:
        return pd.DataFrame(columns=[
            config.COL_LOAN_ID, config.COL_MONTH_INDEX, 'anomaly_score', 
            'top_driver_1', 'top_driver_2', 'top_driver_3', 'driver_values', 'rule_violations'
        ])
        
    # Get top n indices
    top_n_idx = np.argsort(combined_scores)[::-1][:n_actual]
    
    top_df = df.iloc[top_n_idx].copy()
    top_X = X.iloc[top_n_idx][feature_names].fillna(X[feature_names].median())
    
    top_df['anomaly_score'] = combined_scores[top_n_idx]
    
    driver_1_list = []
    driver_2_list = []
    driver_3_list = []
    driver_values_list = []
    
    if model is not None:
        # compute SHAP values for top-n rows
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(top_X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_values = np.abs(shap_values).sum(axis=2)
            
            for i in range(len(top_X)):
                row_shap = np.abs(shap_values[i])
                top_3_idx = np.argsort(row_shap)[::-1][:3]
                top_3_feats = [feature_names[idx] for idx in top_3_idx]
                
                driver_1_list.append(top_3_feats[0] if len(top_3_feats) > 0 else None)
                driver_2_list.append(top_3_feats[1] if len(top_3_feats) > 1 else None)
                driver_3_list.append(top_3_feats[2] if len(top_3_feats) > 2 else None)
                
                vals = [f"{feat}={top_X.iloc[i][feat]}" for feat in top_3_feats]
                driver_values_list.append(", ".join(vals))
                
        except Exception as e:
            logger.warning(f"SHAP explanation failed, fallback to simple differences: {e}")
            model = None # Fallback
            
    if model is None:
        # Fallback to feature values that deviate most from median
        medians = X[feature_names].median()
        stds = X[feature_names].std().replace(0, 1e-9)
        
        for i in range(len(top_X)):
            row = top_X.iloc[i]
            z_scores = np.abs((row - medians) / stds)
            top_3_idx = np.argsort(z_scores.values)[::-1][:3]
            top_3_feats = [feature_names[idx] for idx in top_3_idx]
            
            driver_1_list.append(top_3_feats[0] if len(top_3_feats) > 0 else None)
            driver_2_list.append(top_3_feats[1] if len(top_3_feats) > 1 else None)
            driver_3_list.append(top_3_feats[2] if len(top_3_feats) > 2 else None)
            
            vals = [f"{feat}={row[feat]:.2f} (high/dev)" for feat, idx in zip(top_3_feats, top_3_idx)]
            driver_values_list.append(", ".join(vals))
            
    top_df['top_driver_1'] = driver_1_list
    top_df['top_driver_2'] = driver_2_list
    top_df['top_driver_3'] = driver_3_list
    top_df['driver_values'] = driver_values_list
    
    if 'rule_violation_details' in top_df.columns:
        top_df['rule_violations'] = top_df['rule_violation_details']
    else:
        top_df['rule_violations'] = None
        
    return top_df[[
        config.COL_LOAN_ID, config.COL_MONTH_INDEX, 'anomaly_score', 
        'top_driver_1', 'top_driver_2', 'top_driver_3', 'driver_values', 'rule_violations'
    ]]

def run_anomaly_detection(df: pd.DataFrame, X: pd.DataFrame, feature_names: list,
                         validation_rules: list, xgb_model=None) -> dict:
    """Main entry point.
    Returns: {'if_scores': ..., 'rule_scores': ..., 'combined_scores': ..., 
              'top_anomalies': DataFrame, 'if_model': fitted IsolationForest}"""
    logger.info("Computing Isolation Forest scores...")
    if_scores = compute_anomaly_scores(X, feature_names)
    
    logger.info("Computing Rule Violation scores...")
    rule_df = compute_rule_violations(df, validation_rules)
    rule_scores = rule_df['rule_score'].values
    
    if 'rule_violation_details' not in df.columns:
        df = df.copy()
        df['rule_violation_details'] = rule_df['rule_violation_details'].values
        
    logger.info("Computing Combined scores...")
    combined_scores = compute_combined_anomaly_score(if_scores, rule_scores)
    
    logger.info("Extracting top anomalies...")
    top_anomalies = get_top_anomalies(
        df=df, X=X, combined_scores=combined_scores, 
        model=xgb_model, feature_names=feature_names, n=20
    )
    
    threshold = 0.5
    anomalies_above_threshold = np.sum(combined_scores > threshold)
    print(f"Total anomalies above threshold ({threshold}): {anomalies_above_threshold}")
    print(f"Combined score distribution: mean={combined_scores.mean():.4f}, "
          f"std={combined_scores.std():.4f}, max={combined_scores.max():.4f}, min={combined_scores.min():.4f}")
    
    try:
        if_model = joblib.load(config.MODEL_DIR / 'isolation_forest.joblib')
    except Exception:
        if_model = None

    return {
        'if_scores': if_scores,
        'rule_scores': rule_scores,
        'combined_scores': combined_scores,
        'top_anomalies': top_anomalies,
        'if_model': if_model
    }
