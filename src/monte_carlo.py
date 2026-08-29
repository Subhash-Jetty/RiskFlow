import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import config

def sample_economic_paths(n_sims, macro_scenarios_df):
    shift_cols = ['interest_rate_shift', 'unemployment_shift', 'hpi_shift', 'credit_tightening']
    shifts = macro_scenarios_df[shift_cols]
    mean = shifts.mean()
    cov = shifts.cov()
    rng = np.random.RandomState(config.RANDOM_SEED)
    sampled = rng.multivariate_normal(mean, cov, n_sims)
    
    paths = []
    for row in sampled:
        paths.append(dict(zip(shift_cols, row)))
    return paths

def apply_path_to_features(X, path, feature_names):
    X_mod = X.copy()
    if config.COL_INTEREST_RATE in X_mod.columns:
        X_mod[config.COL_INTEREST_RATE] += path['interest_rate_shift']
    if config.FEAT_RATE_SPREAD in X_mod.columns:
        X_mod[config.FEAT_RATE_SPREAD] += path['interest_rate_shift']
    if config.FEAT_BALANCE_RATIO in X_mod.columns:
        X_mod[config.FEAT_BALANCE_RATIO] *= (1 - path['hpi_shift'] / 100)
    if config.COL_DPD in X_mod.columns:
        X_mod[config.COL_DPD] += path['unemployment_shift'] * 10
    if config.COL_CREDIT_BAND in X_mod.columns:
        X_mod[config.COL_CREDIT_BAND] -= path['credit_tightening']
    return X_mod

def run_monte_carlo(X, models, feature_names, macro_scenarios_df):
    n_sims = config.MONTE_CARLO_N_SIMULATIONS
    paths = sample_economic_paths(n_sims, macro_scenarios_df)
    
    results = []
    for path in paths:
        X_mod = apply_path_to_features(X, path, feature_names)
        X_mod_features = X_mod[feature_names] if feature_names else X_mod
        
        sim_res = {}
        for target in [config.TARGET_NEXT_12M_DEF, config.TARGET_NEXT_3M_DEL, config.TARGET_NEXT_12M_PREPAY]:
            if target in models and 'xgb_calibrated' in models[target]:
                model = models[target]['xgb_calibrated']
                probs = model.predict_proba(X_mod_features)[:, 1]
                sim_res[f'{target}_mean_prob'] = probs.mean()
                if target == config.TARGET_NEXT_12M_DEF and config.COL_ORIG_BALANCE in X_mod.columns:
                    sim_res['portfolio_loss'] = (probs * X_mod[config.COL_ORIG_BALANCE]).mean()
        results.append(sim_res)
        
    df_res = pd.DataFrame(results)
    
    output = {'simulation_results': results}
    
    if f'{config.TARGET_NEXT_12M_DEF}_mean_prob' in df_res.columns:
        output['var_default_95'] = np.percentile(df_res[f'{config.TARGET_NEXT_12M_DEF}_mean_prob'], 95)
        output['var_default_99'] = np.percentile(df_res[f'{config.TARGET_NEXT_12M_DEF}_mean_prob'], 99)
    if 'portfolio_loss' in df_res.columns:
        output['var_loss_95'] = np.percentile(df_res['portfolio_loss'], 95)
        output['var_loss_99'] = np.percentile(df_res['portfolio_loss'], 99)
        
    output['summary_stats'] = df_res.describe().to_dict()
    return output

def plot_monte_carlo_results(results):
    df_res = pd.DataFrame(results['simulation_results'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    def_col = f'{config.TARGET_NEXT_12M_DEF}_mean_prob'
    if def_col in df_res.columns:
        axes[0].hist(df_res[def_col], bins=50, alpha=0.7)
        axes[0].set_title('Default Rate Distribution')
        if 'var_default_95' in results:
            axes[0].axvline(results['var_default_95'], color='r', linestyle='--', label='95% VaR')
        if 'var_default_99' in results:
            axes[0].axvline(results['var_default_99'], color='r', linestyle='-.', label='99% VaR')
        axes[0].legend()
        
    if 'portfolio_loss' in df_res.columns:
        axes[1].hist(df_res['portfolio_loss'], bins=50, alpha=0.7)
        axes[1].set_title('Portfolio Loss Distribution')
        if 'var_loss_95' in results:
            axes[1].axvline(results['var_loss_95'], color='r', linestyle='--', label='95% VaR')
        if 'var_loss_99' in results:
            axes[1].axvline(results['var_loss_99'], color='r', linestyle='-.', label='99% VaR')
        axes[1].legend()
        
    plt.tight_layout()
    plt.savefig(config.REPORT_DIR / 'monte_carlo_loss_distribution.png')
    plt.close()

def generate_monte_carlo_report(results):
    df_res = pd.DataFrame(results['simulation_results'])
    stats = df_res.describe()
    
    md = "# Monte Carlo Simulation Report\n\n"
    md += "## Summary Statistics\n\n"
    md += stats.to_markdown() + "\n\n"
    
    md += "## Value at Risk (VaR)\n\n"
    if 'var_default_95' in results:
        md += f"- **Default Rate 95% VaR**: {results['var_default_95']:.4f}\n"
        md += f"- **Default Rate 99% VaR**: {results['var_default_99']:.4f}\n"
    if 'var_loss_95' in results:
        md += f"- **Portfolio Loss 95% VaR**: {results['var_loss_95']:.4f}\n"
        md += f"- **Portfolio Loss 99% VaR**: {results['var_loss_99']:.4f}\n"
        
    with open(config.REPORT_DIR / 'monte_carlo_report.md', 'w') as f:
        f.write(md)
        
    return md
