import numpy as np
import pandas as pd
from src import config

try:
    import shap
except ImportError:
    shap = None

def find_counterfactual(model, x_original, feature_names, target_threshold=0.5, max_iterations=100, step_size=0.1) -> dict:
    x_val = np.array(x_original).reshape(1, -1)
    orig_prob = model.predict_proba(x_val)[0, 1]
    if orig_prob < target_threshold:
        return None
    
    importances = None
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_val)
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[1][0])
            else:
                shap_vals = np.abs(shap_values[0])
            importances = shap_vals
        except Exception:
            pass
            
    if importances is None:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.ones(len(feature_names))
            
    feat_order = np.argsort(importances)[::-1]
    x_cf = x_val.copy().astype(float)
    changes = []
    success = False
    
    for i in range(max_iterations):
        curr_prob = model.predict_proba(x_cf)[0, 1]
        if curr_prob < target_threshold:
            success = True
            break
            
        feat_idx = feat_order[i % len(feature_names)]
        feat_name = feature_names[feat_idx]
        orig_v = x_val[0, feat_idx]
        curr_v = x_cf[0, feat_idx]
        
        step = step_size * (np.abs(orig_v) if orig_v != 0 else 1.0)
        x_cf[0, feat_idx] -= step
        
        change_idx = -1
        for c_i, c in enumerate(changes):
            if c['feature'] == feat_name:
                change_idx = c_i
                break
                
        if change_idx >= 0:
            changes[change_idx]['new_value'] = x_cf[0, feat_idx]
        else:
            changes.append({
                'feature': feat_name,
                'original_value': orig_v,
                'new_value': x_cf[0, feat_idx],
                'direction': 'decreased'
            })
            
    final_prob = model.predict_proba(x_cf)[0, 1]
    
    return {
        'original_prob': orig_prob,
        'counterfactual_prob': final_prob,
        'changes': changes,
        'success': success
    }

def generate_counterfactual_narrative(cf_result, loan_id) -> str:
    if cf_result is None:
        return f"For loan {loan_id}, no changes needed as probability is below threshold."
        
    orig = cf_result['original_prob']
    new = cf_result['counterfactual_prob']
    changes = cf_result['changes']
    
    parts = []
    for c in changes:
        parts.append(f"{c['feature']} {c['direction']} from {c['original_value']:.2f} to {c['new_value']:.2f}")
        
    changes_str = " and ".join(parts)
    return f"For loan {loan_id}, the default probability would drop from {orig:.2f} to {new:.2f} if {changes_str}."

def run_counterfactual_analysis(df, X, models, feature_names, n_loans=5) -> dict:
    model = models[config.TARGET_NEXT_12M_DEF]['xgb_calibrated']
    probs = model.predict_proba(X)[:, 1]
    
    top_indices = np.argsort(probs)[::-1][:n_loans]
    
    results = {}
    report_lines = ["# Counterfactual Analysis Report\n"]
    
    for idx in top_indices:
        loan_id = df.iloc[idx][config.COL_LOAN_ID]
        x_row = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        
        cf = find_counterfactual(model, x_row, feature_names)
        if cf:
            nar = generate_counterfactual_narrative(cf, loan_id)
            results[loan_id] = cf
            report_lines.append(f"- {nar}")
            
    report_path = config.REPORT_DIR / 'counterfactual_examples.md'
    report_content = "\n".join(report_lines)
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
        
    return {'counterfactuals': results, 'report': report_content}
