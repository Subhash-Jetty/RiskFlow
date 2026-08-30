"""
Explainability module for SHAP, calibration, fairness, errors, and confidence.
"""

import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path

import src.config as config


def compute_shap_values(model, X_sample: pd.DataFrame, feature_names: list) -> tuple:
    """Compute SHAP values using TreeExplainer on a sample of size min(len(X), config.SHAP_SAMPLE_SIZE).
    Returns (shap_values, expected_value, X_sample_used)."""
    sample_size = min(len(X_sample), config.SHAP_SAMPLE_SIZE)
    X_sample_used = X_sample.sample(n=sample_size, random_state=config.RANDOM_SEED)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_used)
    expected_value = explainer.expected_value
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
        
    return shap_values, expected_value, X_sample_used


def generate_global_importance(shap_values, feature_names: list) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature.
    Returns DataFrame: feature, mean_abs_shap, rank. Sorted descending."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    })
    df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


def generate_calibration_analysis(models: dict, X_test, y_test_dict) -> dict:
    """Compute calibration curves for all binary targets.
    For each target with a calibrated model:
    - Compute predicted probabilities
    - Bin into 10 bins
    - Compute fraction of positives per bin
    Saves calibration plot to config.REPORT_DIR / 'calibration_curves.png'.
    Returns dict: {target_name: {'bins': list, 'fraction_pos': list, 'mean_pred': list, 'brier': float}}"""
    results = {}
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    
    for target in config.BINARY_TARGETS:
        if target not in models or 'xgb_calibrated' not in models[target]:
            continue
            
        try:
            model = models[target]['xgb_calibrated']
            y_true = y_test_dict[target]
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
                
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
            brier = brier_score_loss(y_true, y_prob)
            
            results[target] = {
                'bins': 10,
                'fraction_pos': fraction_of_positives.tolist(),
                'mean_pred': mean_predicted_value.tolist(),
                'brier': float(brier)
            }
            
            plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{target} (Brier: {brier:.3f})')
        except Exception as e:
            print(f"Error computing calibration for {target}: {e}")
            
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(config.REPORT_DIR / 'calibration_curves.png')
    plt.close()
    
    return results


def analyze_errors(models: dict, X_test, y_test_dict, feature_names: list) -> dict:
    """Analyze False Positives and False Negatives for binary targets.
    For each target:
    - Identify FP and FN at threshold=0.5
    - Compute summary stats of FP group vs FN group vs correct predictions
    - Identify distinguishing features
    Returns dict: {target_name: {'fp_count': int, 'fn_count': int, 'fp_examples': DataFrame, 'fn_examples': DataFrame,
                                  'fp_feature_profile': dict, 'fn_feature_profile': dict}}"""
    results = {}
    if isinstance(X_test, pd.DataFrame):
        X_df = X_test.copy()
    else:
        X_df = pd.DataFrame(X_test, columns=feature_names)
    
    for target in config.BINARY_TARGETS:
        if target not in models or 'xgb_calibrated' not in models[target]:
            continue
            
        try:
            model = models[target]['xgb_calibrated']
            y_true = y_test_dict[target]
            y_true_arr = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
                
            y_pred = (y_prob >= 0.5).astype(int)
            
            fp_mask = (y_pred == 1) & (y_true_arr == 0)
            fn_mask = (y_pred == 0) & (y_true_arr == 1)
            
            fp_count = fp_mask.sum()
            fn_count = fn_mask.sum()
            
            fp_examples = X_df[fp_mask].head(5)
            fn_examples = X_df[fn_mask].head(5)
            
            fp_profile = X_df[fp_mask].mean().to_dict() if fp_count > 0 else {}
            fn_profile = X_df[fn_mask].mean().to_dict() if fn_count > 0 else {}
            
            results[target] = {
                'fp_count': int(fp_count),
                'fn_count': int(fn_count),
                'fp_examples': fp_examples,
                'fn_examples': fn_examples,
                'fp_feature_profile': fp_profile,
                'fn_feature_profile': fn_profile
            }
        except Exception as e:
            print(f"Error analyzing errors for {target}: {e}")
            
    return results


def compute_fairness_metrics(models: dict, X_test, y_test_dict, df_test, 
                             feature_names: list) -> dict:
    """Compute Demographic Parity across state and credit_score_band.
    For each protected attribute and each binary target:
    - Compute positive prediction rate per group
    - Compute demographic parity difference (max - min rate)
    Returns dict: {target_name: {attribute: {'group_rates': dict, 'dp_diff': float}}}"""
    results = {}
    protected_attrs = [config.COL_STATE, config.COL_CREDIT_BAND]
    
    for target in config.BINARY_TARGETS:
        if target not in models or 'xgb_calibrated' not in models[target]:
            continue
            
        target_results = {}
        try:
            model = models[target]['xgb_calibrated']
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
                
            y_pred = (y_prob >= 0.5).astype(int)
            
            for attr in protected_attrs:
                if attr not in df_test.columns:
                    continue
                    
                attr_values = df_test[attr].values
                temp_df = pd.DataFrame({'attr': attr_values, 'pred': y_pred})
                rates = temp_df.groupby('attr')['pred'].mean().to_dict()
                
                if rates:
                    dp_diff = max(rates.values()) - min(rates.values())
                else:
                    dp_diff = 0.0
                    
                target_results[attr] = {
                    'group_rates': rates,
                    'dp_diff': float(dp_diff)
                }
                
            if target_results:
                results[target] = target_results
        except Exception as e:
            print(f"Error computing fairness for {target}: {e}")
            
    return results


def compute_confidence_metrics(models: dict, X_test) -> dict:
    """Compute prediction confidence/uncertainty metrics.
    For each binary target:
    - Mean predicted probability
    - Std of predicted probabilities
    - % of predictions in uncertain zone (0.3-0.7)
    Returns dict."""
    results = {}
    for target in config.BINARY_TARGETS:
        if target not in models or 'xgb_calibrated' not in models[target]:
            continue
            
        try:
            model = models[target]['xgb_calibrated']
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
                
            mean_prob = float(np.mean(y_prob))
            std_prob = float(np.std(y_prob))
            
            uncertain_mask = (y_prob >= 0.3) & (y_prob <= 0.7)
            uncertain_pct = float(np.mean(uncertain_mask)) * 100
            
            results[target] = {
                'mean_prob': mean_prob,
                'std_prob': std_prob,
                'uncertain_pct': uncertain_pct
            }
        except Exception as e:
            print(f"Error computing confidence for {target}: {e}")
            
    return results


def generate_explainability_report(shap_results: dict, calibration_results: dict,
                                   error_results: dict, fairness_results: dict,
                                   confidence_results: dict, model_metrics: dict) -> str:
    """Generate reports/explainability_report.md.
    ALL numbers computed from input dicts - NEVER hardcode values.
    Returns markdown string."""
    
    target = config.TARGET_NEXT_12M_DEF
    lines = []
    
    lines.append("# Explainability & Fairness Report\n")
    lines.append("This report details the model explainability, calibration, error analysis, fairness, and confidence metrics.\n")
    
    lines.append("## 1. Global Feature Importance (SHAP)")
    lines.append("![SHAP Global Importance](shap_global_importance.png)\n")
    if shap_results and 'global_importance' in shap_results:
        df_imp = shap_results['global_importance']
        lines.append("Top 15 Features:")
        lines.append("| Rank | Feature | Mean Absolute SHAP |")
        lines.append("|---|---|---|")
        for _, row in df_imp.head(15).iterrows():
            lines.append(f"| {row['rank']} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
    lines.append("\n")
    
    lines.append("## 2. Model Calibration")
    lines.append("![Calibration Curves](calibration_curves.png)\n")
    if calibration_results and target in calibration_results:
        brier = calibration_results[target]['brier']
        lines.append(f"For **{target}**, the Brier score is **{brier:.4f}**.\n")
    
    lines.append("## 3. Error Analysis")
    if error_results and target in error_results:
        res = error_results[target]
        lines.append(f"For **{target}** at threshold 0.5:")
        lines.append(f"- False Positives: {res['fp_count']}")
        lines.append(f"- False Negatives: {res['fn_count']}\n")
        
        if res['fp_count'] > 0 and len(res['fp_examples']) > 0:
            lines.append("### False Positive Examples")
            lines.append(res['fp_examples'].to_string())
            lines.append("\n")
            
        if res['fn_count'] > 0 and len(res['fn_examples']) > 0:
            lines.append("### False Negative Examples")
            lines.append(res['fn_examples'].to_string())
            lines.append("\n")
            
    lines.append("## 4. Fairness / Bias Analysis (Demographic Parity)")
    if fairness_results and target in fairness_results:
        lines.append(f"For **{target}**:")
        for attr, fres in fairness_results[target].items():
            dp_diff = fres['dp_diff']
            lines.append(f"### Protected Attribute: `{attr}`")
            lines.append(f"**Demographic Parity Difference:** {dp_diff:.4f}\n")
            
            lines.append("| Group | Positive Prediction Rate |")
            lines.append("|---|---|")
            for grp, rate in fres['group_rates'].items():
                lines.append(f"| {grp} | {rate:.4f} |")
            lines.append("\n")
            
    lines.append("## 5. Prediction Confidence")
    if confidence_results and target in confidence_results:
        res = confidence_results[target]
        lines.append(f"For **{target}**:")
        lines.append(f"- Mean Probability: {res['mean_prob']:.4f}")
        lines.append(f"- Std Dev Probability: {res['std_prob']:.4f}")
        lines.append(f"- % of Predictions in Uncertain Zone (0.3-0.7): {res['uncertain_pct']:.2f}%\n")
        
    return "\n".join(lines)


def generate_segmented_calibration(models: dict, X_test, y_test_dict, df_test) -> dict:
    results = {}
    target = config.TARGET_NEXT_12M_DEF
    
    if target not in models or 'xgb_calibrated' not in models[target]:
        return results
        
    try:
        model = models[target]['xgb_calibrated']
        y_true = y_test_dict[target]
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].plot([0, 1], [0, 1], 'k:')
        axes[1].plot([0, 1], [0, 1], 'k:')
        
        if config.COL_ORIGINATION_MONTH in df_test.columns:
            vintages = pd.to_datetime(df_test[config.COL_ORIGINATION_MONTH]).dt.year
            for year in vintages.unique():
                mask = (vintages == year)
                if mask.sum() > 50:
                    y_t = y_true[mask]
                    y_p = y_prob[mask]
                    fop, mpv = calibration_curve(y_t, y_p, n_bins=10)
                    brier = brier_score_loss(y_t, y_p)
                    axes[0].plot(mpv, fop, 's-', label=f'Vintage {year} (Brier: {brier:.3f})')
                    
        if config.COL_CREDIT_BAND in df_test.columns:
            for band in config.CREDIT_BANDS:
                mask = (df_test[config.COL_CREDIT_BAND] == band)
                if mask.sum() > 50:
                    y_t = y_true[mask]
                    y_p = y_prob[mask]
                    fop, mpv = calibration_curve(y_t, y_p, n_bins=10)
                    brier = brier_score_loss(y_t, y_p)
                    axes[1].plot(mpv, fop, 's-', label=f'Credit {band} (Brier: {brier:.3f})')
                    
        axes[0].set_title('Calibration by Vintage')
        axes[0].legend()
        axes[1].set_title('Calibration by Credit Band')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(config.REPORT_DIR / 'calibration_by_segment.png')
        plt.close()
        
    except Exception as e:
        print(f"Error computing segmented calibration: {e}")
        
    return results


def run_explainability(models: dict, X_train, X_test, y_test_dict, df_test,
                       feature_names: list) -> dict:
    shap_results = {}
    target = config.TARGET_NEXT_12M_DEF
    
    if target in models and 'xgb' in models[target]:
        try:
            model_xgb = models[target]['xgb']
            if isinstance(X_train, pd.DataFrame):
                X_df = X_train.copy()
            else:
                X_df = pd.DataFrame(X_train, columns=feature_names)
            
            shap_values, expected_value, X_sample_used = compute_shap_values(model_xgb, X_df, feature_names)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample_used, plot_type='bar', show=False, feature_names=feature_names)
            plt.tight_layout()
            plt.savefig(config.REPORT_DIR / 'shap_global_importance.png')
            plt.close()
            
            df_imp = generate_global_importance(shap_values, feature_names)
            shap_results['global_importance'] = df_imp
        except Exception as e:
            print(f"Error computing SHAP: {e}")
            
    calib_results = generate_calibration_analysis(models, X_test, y_test_dict)
    segmented_calib = generate_segmented_calibration(models, X_test, y_test_dict, df_test)
    error_results = analyze_errors(models, X_test, y_test_dict, feature_names)
    fairness_results = compute_fairness_metrics(models, X_test, y_test_dict, df_test, feature_names)
    confidence_results = compute_confidence_metrics(models, X_test)
    
    report_md = generate_explainability_report(
        shap_results=shap_results,
        calibration_results=calib_results,
        error_results=error_results,
        fairness_results=fairness_results,
        confidence_results=confidence_results,
        model_metrics={}
    )
    
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EXPLAINABILITY_REPORT, "w") as f:
        f.write(report_md)
        
    return {
        'shap_results': shap_results,
        'calibration_results': calib_results,
        'error_results': error_results,
        'fairness_results': fairness_results,
        'confidence_results': confidence_results
    }
