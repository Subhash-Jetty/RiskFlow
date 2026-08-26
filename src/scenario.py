import pandas as pd
import numpy as np
from pathlib import Path
from src import config

def load_scenarios() -> pd.DataFrame:
    """Load macro_scenarios.csv from config.MACRO_SCENARIOS_FILE.
    Expected columns: scenario_name, interest_rate_shift, unemployment_shift, hpi_shift, credit_tightening.
    Returns DataFrame."""
    try:
        return pd.read_csv(config.MACRO_SCENARIOS_FILE)
    except Exception as e:
        # Fallback empty df if not present
        return pd.DataFrame(columns=[
            'scenario_name', 'interest_rate_shift', 'unemployment_shift', 
            'hpi_shift', 'credit_tightening'
        ])

def apply_scenario(df: pd.DataFrame, X: pd.DataFrame, scenario: pd.Series, 
                   feature_names: list) -> pd.DataFrame:
    """Apply a scenario's shifts to the feature matrix.
    - interest_rate_shift: add to interest_rate and rate_spread features
    - credit_tightening: shift credit bands worse by this many levels
    - hpi_shift: adjust balance_ratio by -hpi_shift/100
    Returns modified X copy."""
    X_stress = X.copy()
    
    # 1. Interest Rate Shift
    ir_shift = scenario.get('interest_rate_shift', 0.0)
    if pd.notna(ir_shift) and ir_shift != 0:
        if config.COL_INTEREST_RATE in X_stress.columns:
            X_stress[config.COL_INTEREST_RATE] += ir_shift
        if config.FEAT_RATE_SPREAD in X_stress.columns:
            X_stress[config.FEAT_RATE_SPREAD] += ir_shift
            
    # 2. Credit Tightening
    ct_shift = scenario.get('credit_tightening', 0)
    if pd.notna(ct_shift) and ct_shift != 0:
        if config.COL_CREDIT_BAND in X_stress.columns:
            # assuming numeric encoding where lower is worse or similar
            X_stress[config.COL_CREDIT_BAND] -= ct_shift
            
    # 3. HPI Shift
    hpi_shift = scenario.get('hpi_shift', 0.0)
    if pd.notna(hpi_shift) and hpi_shift != 0:
        if config.FEAT_BALANCE_RATIO in X_stress.columns:
            X_stress[config.FEAT_BALANCE_RATIO] *= (1.0 - hpi_shift / 100.0)
            
    # 4. Unemployment Shift
    unemp_shift = scenario.get('unemployment_shift', 0.0)
    if pd.notna(unemp_shift) and unemp_shift != 0:
        if config.COL_DPD in X_stress.columns:
            X_stress[config.COL_DPD] += unemp_shift * 10.0
            
    return X_stress

def project_scenario_impacts(df: pd.DataFrame, X: pd.DataFrame, models: dict,
                             scenarios: pd.DataFrame, feature_names: list) -> dict:
    """Run all scenarios through models.
    For each scenario, apply shifts and re-score.
    Returns dict: {scenario_name: {'delinquency_3m_rate': float, 'delinquency_6m_rate': float,
                                    'default_12m_rate': float, 'prepayment_12m_rate': float,
                                    'segment_impacts': DataFrame}}"""
    results = {}
    
    targets = {
        config.TARGET_NEXT_3M_DEL: 'delinquency_3m_rate',
        config.TARGET_NEXT_6M_DEL: 'delinquency_6m_rate',
        config.TARGET_NEXT_12M_DEF: 'default_12m_rate',
        config.TARGET_NEXT_12M_PREPAY: 'prepayment_12m_rate'
    }
    
    # Base predictions
    base_preds = {}
    for target_name in targets.keys():
        if target_name in models:
            model_info = models[target_name]
            model = model_info.get('xgb_calibrated') or model_info.get('xgb') or list(model_info.values())[0]
            try:
                preds = model.predict_proba(X[feature_names])[:, 1]
                base_preds[target_name] = preds
            except Exception:
                pass
                
    if scenarios.empty:
        return results
        
    for _, scenario in scenarios.iterrows():
        scenario_name = scenario.get('scenario_name', f"Scenario_{_}")
        X_stress = apply_scenario(df, X, scenario, feature_names)
        
        scenario_result = {}
        stress_preds = {}
        for target_name, result_key in targets.items():
            if target_name in base_preds:
                model_info = models[target_name]
                model = model_info.get('xgb_calibrated') or model_info.get('xgb') or list(model_info.values())[0]
                try:
                    preds = model.predict_proba(X_stress[feature_names])[:, 1]
                    stress_preds[target_name] = preds
                    scenario_result[result_key] = float(np.mean(preds))
                except Exception:
                    scenario_result[result_key] = np.nan
        
        # Segment impacts
        try:
            scenario_result['segment_impacts'] = compute_segment_impacts(df, base_preds, stress_preds)
        except Exception:
            scenario_result['segment_impacts'] = pd.DataFrame()
            
        results[scenario_name] = scenario_result
        
    return results

def compute_segment_impacts(df: pd.DataFrame, preds_base: dict, preds_stress: dict) -> pd.DataFrame:
    """Compare base vs stressed predictions across segments.
    Segments: credit_score_band, state, servicer_name (vintage approximated by origination_month year).
    Returns DataFrame: segment_type, segment_value, metric, base_rate, stressed_rate, delta."""
    records = []
    
    # Derive vintage year if possible
    vintage = None
    if config.COL_ORIGINATION_MONTH in df.columns:
        try:
            vintage = pd.to_datetime(df[config.COL_ORIGINATION_MONTH]).dt.year
        except Exception:
            vintage = pd.Series([np.nan]*len(df), index=df.index)
    else:
        vintage = pd.Series([np.nan]*len(df), index=df.index)
        
    segment_cols = {
        'credit_score_band': df.get(config.COL_CREDIT_BAND),
        'state': df.get(config.COL_STATE),
        'servicer_name': df.get(config.COL_SERVICER),
        'vintage_year': vintage
    }
    
    for seg_type, seg_series in segment_cols.items():
        if seg_series is None or seg_series.isna().all():
            continue
            
        for seg_val, group_idx in seg_series.groupby(seg_series).groups.items():
            for target_name in preds_base.keys():
                if target_name in preds_stress:
                    b_pred = preds_base[target_name][group_idx]
                    s_pred = preds_stress[target_name][group_idx]
                    
                    b_mean = float(np.mean(b_pred))
                    s_mean = float(np.mean(s_pred))
                    delta = s_mean - b_mean
                    
                    records.append({
                        'segment_type': seg_type,
                        'segment_value': str(seg_val),
                        'metric': target_name,
                        'base_rate': b_mean,
                        'stressed_rate': s_mean,
                        'delta': delta
                    })
                    
    return pd.DataFrame(records)

def identify_top_drivers(base_preds: dict, stress_preds: dict, feature_names: list,
                         X_base: pd.DataFrame, X_stress: pd.DataFrame) -> list:
    """Identify which features/segments drove the biggest movement.
    Returns list of (feature_name, avg_shift, impact_on_prediction) tuples."""
    drivers = []
    
    if len(X_base) == 0:
        return drivers

    impacts = np.zeros(len(X_base))
    count = 0
    for target in base_preds:
        if target in stress_preds:
            impacts += (stress_preds[target] - base_preds[target])
            count += 1
            
    if count == 0:
        return drivers
        
    impacts /= count
    
    for feat in feature_names:
        if feat in X_base.columns and feat in X_stress.columns:
            diff = X_stress[feat] - X_base[feat]
            if diff.any():
                avg_shift = diff.mean()
                if diff.std() > 0 and pd.Series(impacts).std() > 0:
                    corr = np.corrcoef(diff, impacts)[0, 1]
                    if pd.notna(corr):
                        impact_on_pred = corr * np.mean(np.abs(impacts))
                    else:
                        impact_on_pred = 0.0
                else:
                    impact_on_pred = 0.0
                drivers.append((feat, float(avg_shift), float(impact_on_pred)))
                
    # Sort by absolute impact
    drivers.sort(key=lambda x: abs(x[2]), reverse=True)
    return drivers

def generate_scenario_report(scenario_results: dict) -> str:
    """Generate reports/scenario_report.md content from computed results.
    ALL numbers must come from scenario_results dict - never hardcode.
    Returns markdown string."""
    report = ["# Macro Scenario Stress Testing Report\n"]
    
    if not scenario_results:
        report.append("No scenarios were evaluated.\n")
        
    for scenario_name, results in scenario_results.items():
        report.append(f"## Scenario: {scenario_name}\n")
        
        report.append("### Portfolio-Level Impacts")
        report.append("| Metric | Stressed Rate |")
        report.append("|--------|---------------|")
        for m in ['delinquency_3m_rate', 'delinquency_6m_rate', 'default_12m_rate', 'prepayment_12m_rate']:
            if m in results:
                val = results[m]
                if not np.isnan(val) and not isinstance(val, pd.DataFrame):
                    report.append(f"| {m} | {val:.4f} |")
        report.append("\n")
        
        report.append("### Top 10 Most Impacted Segments")
        report.append("| Segment Type | Segment Value | Metric | Base Rate | Stressed Rate | Delta |")
        report.append("|--------------|---------------|--------|-----------|---------------|-------|")
        
        seg_df = results.get('segment_impacts', pd.DataFrame())
        if not seg_df.empty:
            # top 10 by absolute delta
            top_segs = seg_df.assign(abs_delta=seg_df['delta'].abs()).sort_values('abs_delta', ascending=False).head(10)
            for _, row in top_segs.iterrows():
                report.append(f"| {row['segment_type']} | {row['segment_value']} | {row['metric']} | {row['base_rate']:.4f} | {row['stressed_rate']:.4f} | {row['delta']:.4f} |")
        else:
            report.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
        report.append("\n")
        
    report_content = "\n".join(report)
    
    config.SCENARIO_REPORT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config.SCENARIO_REPORT, 'w') as f:
            f.write(report_content)
    except Exception:
        pass
        
    return report_content

def run_scenario_analysis(df: pd.DataFrame, X: pd.DataFrame, models: dict, 
                          feature_names: list) -> dict:
    """Main entry point.
    Returns: {'scenarios': DataFrame, 'results': dict, 'report': str}"""
    scenarios = load_scenarios()
    results = project_scenario_impacts(df, X, models, scenarios, feature_names)
    report = generate_scenario_report(results)
    
    return {
        'scenarios': scenarios,
        'results': results,
        'report': report
    }
