import time
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from src import config

def run_single_experiment(config_dict, X_train, y_train, X_val, y_val, experiment_id):
    start_time = time.time()
    
    scale_pos_weight = 1.0
    num_pos = sum(y_train == 1)
    if num_pos > 0:
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos
        
    model = xgb.XGBClassifier(
        n_estimators=config_dict.get('n_estimators', 100),
        max_depth=config_dict.get('max_depth', 6),
        learning_rate=config_dict.get('learning_rate', 0.1),
        subsample=config_dict.get('subsample', 0.8),
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=30
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_val, probs),
        'pr_auc': average_precision_score(y_val, probs),
        'f1': f1_score(y_val, preds),
        'brier_score': brier_score_loss(y_val, probs)
    }
    
    training_time = time.time() - start_time
    
    return {
        'experiment_id': experiment_id,
        'config': config_dict,
        'metrics': metrics,
        'training_time': training_time
    }

def run_experiment_grid(X_train, y_train, X_val, y_val):
    results = []
    
    for i, config_dict in enumerate(config.EXPERIMENT_GRID):
        res = run_single_experiment(config_dict, X_train, y_train, X_val, y_val, i)
        results.append(res)
        
        try:
            import mlflow
            with mlflow.start_run(run_name=f"grid_search_{i}"):
                mlflow.log_params(config_dict)
                mlflow.log_metrics(res['metrics'])
        except ImportError:
            pass
            
    best_exp = max(results, key=lambda x: x['metrics']['roc_auc'])
    
    return {
        'experiments': results,
        'best_experiment': best_exp,
        'best_config': best_exp['config']
    }

def save_experiment_log(results):
    with open(config.EXPERIMENT_LOG_FILE, 'w') as f:
        json.dump(results, f, indent=4)

def generate_experiment_report(results):
    lines = [
        "# Agentic Experiment Runner Report",
        "",
        "| ID | n_estimators | max_depth | learning_rate | subsample | ROC-AUC | F1 Score | Brier Score | Time (s) |",
        "|----|-------------|-----------|---------------|-----------|---------|----------|-------------|----------|"
    ]
    
    for exp in results['experiments']:
        cfg = exp['config']
        met = exp['metrics']
        best_marker = "*" if exp['experiment_id'] == results['best_experiment']['experiment_id'] else ""
        lines.append(
            f"| {exp['experiment_id']} {best_marker} | {cfg.get('n_estimators')} | {cfg.get('max_depth')} | {cfg.get('learning_rate')} | {cfg.get('subsample')} | "
            f"{met['roc_auc']:.4f} | {met['f1']:.4f} | {met['brier_score']:.4f} | {exp['training_time']:.1f} |"
        )
        
    md = "\n".join(lines)
    with open(config.REPORT_DIR / 'experiment_report.md', 'w', encoding='utf-8') as f:
        f.write(md)
        
    return md

def run_agentic_experiments(X_train, y_train_dict, X_val, y_val_dict):
    target = config.TARGET_NEXT_12M_DEF
    y_train = y_train_dict[target]
    y_val = y_val_dict[target]
    
    mask_train = ~y_train.isna()
    mask_val = ~y_val.isna()
    
    results = run_experiment_grid(X_train[mask_train], y_train[mask_train], X_val[mask_val], y_val[mask_val])
    save_experiment_log(results)
    generate_experiment_report(results)
    
    return results
