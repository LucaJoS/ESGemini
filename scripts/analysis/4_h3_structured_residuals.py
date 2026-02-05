"""
Script 4: Structured Residuals (H3)

Purpose:
    Tests whether forecast errors are systematically related to observable factors.
    DV: Absolute forecast error (|prediction - NRS|)
    IVs: DaysSinceUpdate, Controversies, Certainty, DocumentLength

Input:
    - results/data/working_data.db (period_dataset)

Output:
    - results/csv/h3_regression_results.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config, utils

print("Script 4: Structured Residuals (H3)")
print("-" * 60)

# Load data
df_periods = utils.load_working_data('period_dataset')
print(f"Total periods: {len(df_periods)}")

results_all = {}

for model in config.ALL_MODELS:
    print(f"\n[{model}]")
    
    # Get rolling average columns
    simple_col = config.get_rol_avg_col(model, 'simple')
    cert_col = config.get_rol_avg_col(model, 'cert')
    length_col = config.get_rol_avg_col(model, 'length')
    
    # Calculate errors
    df_periods[f'{model}_error_simple'] = df_periods[simple_col] - df_periods['NRS']
    df_periods[f'{model}_error_cert'] = df_periods[cert_col] - df_periods['NRS']
    df_periods[f'{model}_error_length'] = df_periods[length_col] - df_periods['NRS']
    
    # Absolute errors
    df_periods[f'{model}_abserr_simple'] = np.abs(df_periods[f'{model}_error_simple'])
    df_periods[f'{model}_abserr_cert'] = np.abs(df_periods[f'{model}_error_cert'])
    df_periods[f'{model}_abserr_length'] = np.abs(df_periods[f'{model}_error_length'])
    
    # -------------------------------------------------------------------------
    # Regression 1: Period characteristics -> Forecast error
    # -------------------------------------------------------------------------
    
    # Get certainty column (database uses 'con' for anchored models)
    model_cert_col = {
        'base': 'base_cert_itself_notrol_avg',
        'base_a': 'base_con_cert_itself_notrol_avg', 
        'ft': 'ft_cert_itself_notrol_avg',
        'ft_a': 'ft_con_cert_itself_notrol_avg'
    }.get(model, None) if 'base_cert_itself_notrol_avg' in df_periods.columns else None
    
    # Build regression data
    reg_cols = [f'{model}_abserr_simple', 'period_length_days', 'controversies_score', 'supersector', 'year']
    if model_cert_col and model_cert_col in df_periods.columns:
        reg_cols.append(model_cert_col)
    if 'report_length_itself_notrol_avg' in df_periods.columns:
        reg_cols.append('report_length_itself_notrol_avg')
    
    df_reg = df_periods[reg_cols].dropna()
    
    if len(df_reg) == 0:
        print(f"    No complete data for regression")
        results_all[f'{model}_reg1'] = {'n': 0, 'error': 'No complete data'}
        continue
        
    print(f"    Regression N={len(df_reg)}")
    
    y = df_reg[f'{model}_abserr_simple']
    
    x_cols = ['period_length_days', 'controversies_score']
    if model_cert_col and model_cert_col in df_reg.columns:
        x_cols.append(model_cert_col)
    if 'report_length_itself_notrol_avg' in df_reg.columns:
        x_cols.append('report_length_itself_notrol_avg')
    
    X = sm.add_constant(df_reg[x_cols])
    
    try:
        model_ols = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_reg['supersector']})
        
        results_all[f'{model}_reg1'] = {
            'n': len(df_reg),
            'r2': model_ols.rsquared,
            'period_length_coef': model_ols.params.get('period_length_days', np.nan),
            'period_length_pval': model_ols.pvalues.get('period_length_days', np.nan),
            'controversies_coef': model_ols.params.get('controversies_score', np.nan),
            'controversies_pval': model_ols.pvalues.get('controversies_score', np.nan),
            'certainty_coef': model_ols.params.get(model_cert_col, np.nan) if model_cert_col else np.nan,
            'certainty_pval': model_ols.pvalues.get(model_cert_col, np.nan) if model_cert_col else np.nan,
            'length_coef': model_ols.params.get('report_length_itself_notrol_avg', np.nan),
            'length_pval': model_ols.pvalues.get('report_length_itself_notrol_avg', np.nan)
        }
        print(f"    R2={model_ols.rsquared:.3f}")
    except Exception as e:
        print(f"    Regression failed: {e}")
        results_all[f'{model}_reg1'] = {'n': len(df_reg), 'error': str(e)}
    
    # -------------------------------------------------------------------------
    # Regression 2: Compare rolling average types
    # -------------------------------------------------------------------------
    df_reg2 = df_periods[[f'{model}_abserr_simple', f'{model}_abserr_cert', 
                          f'{model}_abserr_length']].dropna()
    
    if len(df_reg2) > 0:
        mae_simple = df_reg2[f'{model}_abserr_simple'].mean()
        mae_cert = df_reg2[f'{model}_abserr_cert'].mean()
        mae_length = df_reg2[f'{model}_abserr_length'].mean()
        
        t_stat, p_val = stats.ttest_rel(df_reg2[f'{model}_abserr_simple'], 
                                         df_reg2[f'{model}_abserr_cert'])
        
        results_all[f'{model}_improvement_cert'] = {
            'mae_reduction': mae_simple - mae_cert,
            't_stat': t_stat,
            'p_val': p_val
        }
        print(f"    MAE: simple={mae_simple:.2f}, cert={mae_cert:.2f}, length={mae_length:.2f}")

# Save results
df_h3 = pd.DataFrame(results_all).T
utils.save_table_csv(df_h3, 'h3_regression_results')

print("\n" + "-" * 60)
print("H3 complete")
print("Saved: h3_regression_results.csv")
