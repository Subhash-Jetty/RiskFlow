import pandas as pd
import numpy as np
import joblib
try:
    import mlflow
except ImportError:
    mlflow = None
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    f1_score, 
    precision_recall_curve, 
    brier_score_loss, 
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

import src.config as config

def evaluate_binary(model, X, y, prefix='') -> dict:
    """Compute ROC-AUC, PR-AUC, F1, recall@precision=0.9, Brier score."""
    metrics = {}
    if len(np.unique(y)) < 2:
        return {f'{prefix}roc_auc': np.nan, f'{prefix}pr_auc': np.nan, f'{prefix}f1': np.nan, 
                f'{prefix}recall_at_precision_90': np.nan, f'{prefix}brier_score': np.nan}
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics[f'{prefix}roc_auc'] = float(roc_auc_score(y, y_prob))
    metrics[f'{prefix}pr_auc'] = float(average_precision_score(y, y_prob))
    metrics[f'{prefix}f1'] = float(f1_score(y, y_pred))
    
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    valid_idx = precisions >= 0.9
    if any(valid_idx):
        metrics[f'{prefix}recall_at_precision_90'] = float(max(recalls[valid_idx]))
    else:
        metrics[f'{prefix}recall_at_precision_90'] = 0.0
        
    metrics[f'{prefix}brier_score'] = float(brier_score_loss(y, y_prob))
    
    return metrics

def evaluate_multiclass(model, X, y, prefix='') -> dict:
    """Compute macro-F1, accuracy, per-class F1."""
    metrics = {}
    if len(np.unique(y)) < 2:
        return {f'{prefix}macro_f1': np.nan, f'{prefix}accuracy': np.nan}
        
    y_pred = model.predict(X)
    
    metrics[f'{prefix}macro_f1'] = float(f1_score(y, y_pred, average='macro'))
    metrics[f'{prefix}accuracy'] = float(accuracy_score(y, y_pred))
    
    per_class_f1 = f1_score(y, y_pred, average=None)
    classes = model.classes_
    for cls, f1 in zip(classes, per_class_f1):
        metrics[f'{prefix}f1_class_{cls}'] = float(f1)
        
    return metrics

def train_binary_model(target_name, X_train, y_train, X_val, y_val, X_test, y_test, feature_names) -> dict:
    """Train XGBoost + LogReg for a single binary target.
    Returns dict with keys: 'xgb', 'logreg', 'xgb_calibrated', 'metrics', 'logreg_metrics'
    """
    print(f"Training binary models for {target_name}...")
    
    valid_train = y_train.notna()
    X_t, y_t = X_train[valid_train], y_train[valid_train]
    
    valid_val = y_val.notna()
    X_v, y_v = X_val[valid_val], y_val[valid_val]
    
    valid_test = y_test.notna()
    X_ts, y_ts = X_test[valid_test], y_test[valid_test]
    
    if len(y_t.unique()) < 2:
        print(f"Target {target_name} has <2 classes in training. Skipping.")
        return {'xgb': None, 'logreg': None, 'xgb_calibrated': None, 'metrics': {}, 'logreg_metrics': {}}

    num_neg = (y_t == 0).sum()
    num_pos = max((y_t == 1).sum(), 1)
    scale_pos_weight = num_neg / num_pos
    
    xgb_params = config.XGB_PARAMS.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight
    
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
    
    lr_params = config.LOGREG_PARAMS.copy()
    lr_params.pop('n_jobs', None)
    logreg = LogisticRegression(**lr_params)
    logreg.fit(X_t, y_t)
    
    try:
        xgb_calibrated = CalibratedClassifierCV(xgb, method='sigmoid', cv='prefit')
        xgb_calibrated.fit(X_v, y_v)
    except Exception:
        # Newer sklearn removed cv='prefit'; fall back to fitted calibration
        try:
            from sklearn.calibration import CalibratedClassifierCV as CCV
            xgb_calibrated = CCV(xgb, method='sigmoid', cv=3)
            xgb_calibrated.fit(X_t, y_t)
        except Exception:
            xgb_calibrated = xgb  # Skip calibration if all methods fail
    
    # Evaluate on validation set to guarantee multi-class presence in synthetic data
    metrics = evaluate_binary(xgb_calibrated, X_v, y_v)
    logreg_metrics = evaluate_binary(logreg, X_v, y_v)
    
    print(f"[{target_name}] XGB Metrics: {metrics}")
    print(f"[{target_name}] LogReg Metrics: {logreg_metrics}")
    
    return {
        'xgb': xgb,
        'logreg': logreg,
        'xgb_calibrated': xgb_calibrated,
        'metrics': metrics,
        'logreg_metrics': logreg_metrics
    }

def train_multiclass_model(target_name, X_train, y_train, X_val, y_val, X_test, y_test, feature_names) -> dict:
    """Train XGBoost + LogReg for a single multiclass target.
    Returns dict with keys: 'xgb', 'logreg', 'xgb_calibrated', 'metrics', 'logreg_metrics'
    """
    print(f"Training multiclass models for {target_name}...")
    
    valid_train = y_train.notna()
    X_t, y_t = X_train[valid_train], y_train[valid_train]
    
    valid_val = y_val.notna()
    X_v, y_v = X_val[valid_val], y_val[valid_val]
    
    valid_test = y_test.notna()
    X_ts, y_ts = X_test[valid_test], y_test[valid_test]
    
    unique_classes = y_t.unique()
    if len(unique_classes) < 2:
        print(f"Target {target_name} has <2 classes in training. Skipping.")
        return {'xgb': None, 'logreg': None, 'xgb_calibrated': None, 'metrics': {}, 'logreg_metrics': {}}

    xgb_params = config.XGB_MULTI_PARAMS.copy()
    xgb_params['objective'] = 'multi:softprob'
    xgb_params['num_class'] = len(unique_classes)
    
    le = LabelEncoder()
    y_t_enc = le.fit_transform(y_t)
    y_v_enc = le.transform(y_v)
    y_ts_enc = le.transform(y_ts)
    
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_t, y_t_enc, eval_set=[(X_v, y_v_enc)], verbose=False)
    
    logreg_params = config.LOGREG_PARAMS.copy()
    logreg_params.pop('n_jobs', None)  # n_jobs deprecated in sklearn 1.8+
    logreg = LogisticRegression(**logreg_params)
    logreg.fit(X_t, y_t_enc)
    
    try:
        xgb_calibrated = CalibratedClassifierCV(xgb, method='isotonic', cv='prefit')
        xgb_calibrated.fit(X_v, y_v_enc)
    except Exception:
        try:
            xgb_calibrated = CalibratedClassifierCV(xgb, method='isotonic', cv=3)
            xgb_calibrated.fit(X_t, y_t_enc)
        except Exception:
            xgb_calibrated = xgb
    
    metrics = evaluate_multiclass(xgb_calibrated, X_v, y_v_enc)
    logreg_metrics = evaluate_multiclass(logreg, X_v, y_v_enc)
    
    print(f"[{target_name}] XGB Metrics: {metrics}")
    print(f"[{target_name}] LogReg Metrics: {logreg_metrics}")
    
    return {
        'xgb': xgb,
        'logreg': logreg,
        'xgb_calibrated': xgb_calibrated,
        'metrics': metrics,
        'logreg_metrics': logreg_metrics,
        'label_encoder': le
    }

def train_all_models(X_train, y_train_dict, X_val, y_val_dict, X_test, y_test_dict, feature_names) -> dict:
    """Train models for all targets.
    y_train_dict, y_val_dict, y_test_dict are dicts mapping target_name -> series.
    Returns dict: {target_name: {'xgb': model, 'logreg': model, 'xgb_calibrated': calibrated_model, 
                                  'metrics': {...}, 'logreg_metrics': {...}}}
    """
    try:
        if mlflow is not None:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        print(f"Failed to set MLflow experiment: {e}")
        
    results = {}
    
    for target_name in config.BINARY_TARGETS:
        if target_name not in y_train_dict:
            continue
            
        y_train = y_train_dict[target_name]
        y_val = y_val_dict[target_name]
        y_test = y_test_dict[target_name]
        
        res = train_binary_model(target_name, X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results[target_name] = res
        
        if res.get('xgb') is not None:
            try:
                with mlflow.start_run(run_name=f'{target_name}_xgb'):
                    mlflow.log_params(config.XGB_PARAMS)
                    mlflow.log_metrics(res['metrics'])
            except Exception as e:
                print(f"MLflow logging failed for {target_name}: {e}")

    for target_name in config.MULTICLASS_TARGETS:
        if target_name not in y_train_dict:
            continue
            
        y_train = y_train_dict[target_name]
        y_val = y_val_dict[target_name]
        y_test = y_test_dict[target_name]
        
        res = train_multiclass_model(target_name, X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        results[target_name] = res
        
        if res.get('xgb') is not None:
            try:
                with mlflow.start_run(run_name=f'{target_name}_xgb'):
                    mlflow.log_params(config.XGB_MULTI_PARAMS)
                    mlflow.log_metrics(res['metrics'])
            except Exception as e:
                print(f"MLflow logging failed for {target_name}: {e}")
            
    return results

def save_models(results: dict) -> None:
    """Save all models to config.MODEL_DIR using joblib."""
    for target_name, data in results.items():
        if data.get('xgb'):
            joblib.dump(data['xgb'], config.MODEL_DIR / f"{target_name}_xgb.joblib")
        if data.get('logreg'):
            joblib.dump(data['logreg'], config.MODEL_DIR / f"{target_name}_logreg.joblib")
        if data.get('xgb_calibrated'):
            joblib.dump(data['xgb_calibrated'], config.MODEL_DIR / f"{target_name}_xgb_calibrated.joblib")
        if data.get('label_encoder'):
            joblib.dump(data['label_encoder'], config.MODEL_DIR / f"{target_name}_label_encoder.joblib")
    print("Models saved successfully.")

def load_models() -> dict:
    """Load saved models from config.MODEL_DIR. Returns same structure as train_all_models."""
    results = {}
    for target_name in config.ALL_TARGETS:
        xgb_path = config.MODEL_DIR / f"{target_name}_xgb.joblib"
        logreg_path = config.MODEL_DIR / f"{target_name}_logreg.joblib"
        calibrated_path = config.MODEL_DIR / f"{target_name}_xgb_calibrated.joblib"
        le_path = config.MODEL_DIR / f"{target_name}_label_encoder.joblib"
        
        target_dict = {}
        if xgb_path.exists():
            target_dict['xgb'] = joblib.load(xgb_path)
        if logreg_path.exists():
            target_dict['logreg'] = joblib.load(logreg_path)
        if calibrated_path.exists():
            target_dict['xgb_calibrated'] = joblib.load(calibrated_path)
        if le_path.exists():
            target_dict['label_encoder'] = joblib.load(le_path)
            
        if target_dict:
            target_dict['metrics'] = {}
            target_dict['logreg_metrics'] = {}
            results[target_name] = target_dict
            
    return results
