import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from src import config

def compute_feature_clusters(X, feature_names, threshold=None):
    threshold = threshold if threshold is not None else config.CLUSTER_CORR_THRESHOLD
    numeric_features = [f for f in feature_names if pd.api.types.is_numeric_dtype(X[f])]
    X_num = X[numeric_features]
    corr = X_num.corr().abs()
    distance = 1 - corr
    dist_array = squareform(distance, checks=False)
    Z = hierarchy.linkage(dist_array, method='complete')
    labels = hierarchy.fcluster(Z, t=1 - threshold, criterion='distance')
    
    clusters = {}
    for feature, label in zip(numeric_features, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(feature)
        
    return list(clusters.values())

def stress_test_cluster(X, cluster_features, models, feature_names, direction='up', n_sigma=1.0):
    X_stress = X.copy()
    
    for f in cluster_features:
        std_val = X[f].std()
        if direction == 'up':
            X_stress[f] = X_stress[f] + n_sigma * std_val
        elif direction == 'down':
            X_stress[f] = X_stress[f] - n_sigma * std_val
            
    res = {
        'cluster_features': cluster_features,
        'direction': direction,
        'mean_default_rate_delta': 0.0,
        'mean_delinquency_rate_delta': 0.0
    }
    
    for target, model_dict in models.items():
        if target not in [config.TARGET_NEXT_12M_DEF, config.TARGET_NEXT_3M_DEL] or 'xgb_calibrated' not in model_dict:
            continue
        model = model_dict['xgb_calibrated']
        base_pred = model.predict_proba(X[feature_names])[:, 1]
        stress_pred = model.predict_proba(X_stress[feature_names])[:, 1]
        delta = (stress_pred.mean() - base_pred.mean())
        if target == config.TARGET_NEXT_12M_DEF:
            res['mean_default_rate_delta'] = delta
        elif target == config.TARGET_NEXT_3M_DEL:
            res['mean_delinquency_rate_delta'] = delta
            
    return res

def run_stress_sensitivity(X, models, feature_names):
    clusters = compute_feature_clusters(X, feature_names)
    results = []
    
    for cluster in clusters:
        res_up = stress_test_cluster(X, cluster, models, feature_names, 'up', 1.0)
        res_down = stress_test_cluster(X, cluster, models, feature_names, 'down', 1.0)
        results.extend([res_up, res_down])
        
    results.sort(key=lambda x: max(abs(x.get('mean_default_rate_delta', 0)), abs(x.get('mean_delinquency_rate_delta', 0))), reverse=True)
    
    report_lines = []
    report_lines.append("# Stress Sensitivity by Feature Cluster\n")
    report_lines.append("| Cluster Features | Direction | Default Rate Delta | Delinquency Rate Delta |")
    report_lines.append("|---|---|---|---|")
    
    for r in results:
        features_str = ", ".join(r['cluster_features'])
        d1 = r.get('mean_default_rate_delta', 0)
        d2 = r.get('mean_delinquency_rate_delta', 0)
        report_lines.append(f"| {features_str} | {r['direction']} | {d1:.4f} | {d2:.4f} |")
        
    report = "\n".join(report_lines)
    
    out_path = config.REPORT_DIR / 'stress_sensitivity_clusters.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(report)
        
    return {
        'clusters': clusters,
        'stress_results': results,
        'report': report
    }
