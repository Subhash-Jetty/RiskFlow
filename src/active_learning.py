import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from src import config

def identify_uncertain_samples(model, X, df, n=None):
    if n is None:
        n = config.ACTIVE_LEARNING_BATCH_SIZE
    probs = model.predict_proba(X)[:, 1]
    lower, upper = config.ACTIVE_LEARNING_UNCERTAINTY_BAND
    mask = (probs >= lower) & (probs <= upper)
    uncertain_idx = np.where(mask)[0]
    
    uncertain_df = df.iloc[uncertain_idx].copy()
    uncertain_df['original_position_idx'] = uncertain_idx
    uncertain_df['predicted_prob'] = probs[uncertain_idx]
    uncertain_df['distance_to_half'] = np.abs(uncertain_df['predicted_prob'] - 0.5)
    
    uncertain_df = uncertain_df.sort_values('distance_to_half').head(n).copy()
    uncertain_df['uncertainty_rank'] = np.arange(1, len(uncertain_df) + 1)
    
    uncertain_df['index'] = uncertain_df.index
    
    return uncertain_df[['index', 'original_position_idx', config.COL_LOAN_ID, 'predicted_prob', 'uncertainty_rank']]

def simulate_feedback(uncertain_df, y_true, noise_rate=0.05):
    rng = np.random.RandomState(config.RANDOM_SEED)
    feedback_list = []
    
    # y_true may be a reset-index series, while uncertain_df['index'] refers to the original.
    # Since uncertain_df was sliced from df_train which was NOT reset, but y_train WAS reset, 
    # we should just use the positional integer index of uncertain_df if it matches X_train.
    # Wait, uncertain_idx is the positional integer index from the numpy array where mask was applied!
    # So idx (which came from uncertain_idx) is exactly the integer position we need for y_true.
    
    for row_pos, (_, row) in enumerate(uncertain_df.iterrows()):
        idx = int(row['index'])
        # The 'index' column in uncertain_df holds the original index. But we need the positional index!
        # Oh, uncertain_idx was created from np.where(mask)[0] which is positional.
        # But then uncertain_df['index'] = uncertain_df.index saved the ORIGINAL pandas index.
        # Let's just use the positional index directly.
        # Wait, the positional index wasn't saved in uncertain_df!
        pass
    
    # Let's completely rewrite simulate_feedback to just use the actual labels.
    for _, row in uncertain_df.iterrows():
        idx = int(row['original_position_idx'])
        actual_label = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
        
        if rng.rand() < noise_rate:
            human_label = 1 - actual_label
        else:
            human_label = actual_label
            
        confidence = rng.choice(['high', 'medium'])
        
        feedback_list.append({
            'index': idx,
            config.COL_LOAN_ID: row[config.COL_LOAN_ID],
            'predicted_prob': float(row['predicted_prob']),
            'human_label': int(human_label),
            'reviewer_confidence': confidence
        })
        
    return feedback_list

def save_feedback(feedback, path=None):
    save_path = path if path else config.HUMAN_FEEDBACK_FILE
    with open(save_path, 'w') as f:
        json.dump(feedback, f, indent=4)

def load_feedback(path=None):
    load_path = path if path else config.HUMAN_FEEDBACK_FILE
    with open(load_path, 'r') as f:
        return json.load(f)

def retrain_with_feedback(model, X_train, y_train, feedback, X_val, y_val):
    val_preds_before = model.predict_proba(X_val)[:, 1]
    auc_before = roc_auc_score(y_val, val_preds_before)
    f1_before = f1_score(y_val, (val_preds_before > 0.5).astype(int))
    
    y_train_new = y_train.copy()
    for item in feedback:
        idx = item['index']
        label = item['human_label']
        if isinstance(y_train_new, pd.Series):
            y_train_new.iloc[idx] = label
        else:
            y_train_new[idx] = label
            
    params = model.get_params()
    if 'early_stopping_rounds' in params:
        params.pop('early_stopping_rounds')
        
    new_model = XGBClassifier(**params)
    new_model.fit(X_train, y_train_new)
    
    val_preds_after = new_model.predict_proba(X_val)[:, 1]
    auc_after = roc_auc_score(y_val, val_preds_after)
    f1_after = f1_score(y_val, (val_preds_after > 0.5).astype(int))
    
    return {
        'model_before_metrics': {'roc_auc': auc_before, 'f1': f1_before},
        'model_after_metrics': {'roc_auc': auc_after, 'f1': f1_after},
        'n_feedback_samples': len(feedback),
        'improvement': {
            'roc_auc': auc_after - auc_before,
            'f1': f1_after - f1_before
        },
        'retrained_model': new_model
    }

def run_active_learning(models, X_train, y_train_dict, X_val, y_val_dict, df_train):
    model = models[config.TARGET_NEXT_12M_DEF]['xgb_calibrated']
        
    y_train = y_train_dict[config.TARGET_NEXT_12M_DEF]
    y_val = y_val_dict[config.TARGET_NEXT_12M_DEF]
    
    uncertain_df = identify_uncertain_samples(model, X_train, df_train)
    feedback = simulate_feedback(uncertain_df, y_train)
    save_feedback(feedback)
    
    results = retrain_with_feedback(model, X_train, y_train, feedback, X_val, y_val)
    
    report = [
        "# Active Learning Report\n",
        f"**Feedback Samples:** {results['n_feedback_samples']}\n",
        "## Metrics Before Retraining",
        f"- ROC AUC: {results['model_before_metrics']['roc_auc']:.4f}",
        f"- F1 Score: {results['model_before_metrics']['f1']:.4f}\n",
        "## Metrics After Retraining",
        f"- ROC AUC: {results['model_after_metrics']['roc_auc']:.4f}",
        f"- F1 Score: {results['model_after_metrics']['f1']:.4f}\n",
        "## Improvement",
        f"- ROC AUC: {results['improvement']['roc_auc']:+.4f}",
        f"- F1 Score: {results['improvement']['f1']:+.4f}\n"
    ]
    
    report_path = Path(config.REPORT_DIR) / 'active_learning_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
        
    return {
        'uncertain_samples': uncertain_df,
        'feedback': feedback,
        'retrain_results': results
    }
