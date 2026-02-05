"""
Script 6: Additional Diagnostic Analyses

Purpose:
    Supplementary analyses for model diagnostics including:
    - Certainty-error correlations
    - Direction prediction confusion matrices  
    - Report length effects
    - Temporal performance patterns

Input:
    - results/data/working_data.db (period_dataset)

Output:
    - results/csv/enhancements_certainty_analysis.csv
    - results/csv/enhancements_direction_metrics.csv
    - results/csv/enhancements_temporal_analysis.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config, utils

print("Script 6: Additional Diagnostic Analyses")
print("-" * 60)

df_periods = utils.load_working_data('period_dataset')

# =============================================================================
# 1. Certainty Analysis
# =============================================================================
print("\n[1] Certainty-Error Correlations")

certainty_results = []
for model in config.ALL_MODELS:
    cert_col = config.CERT_COLUMNS[model]
    miss_col = config.get_miss_abs_col(model, 'simple')
    
    valid = df_periods[[cert_col, miss_col]].dropna()
    corr = valid[cert_col].corr(valid[miss_col])
    
    certainty_results.append({
        'model': model,
        'correlation': corr,
        'n_obs': len(valid)
    })
    print(f"    {model}: r={corr:.4f}")

df_certainty = pd.DataFrame(certainty_results)
utils.save_table_csv(df_certainty, 'enhancements_certainty_analysis')

# =============================================================================
# 2. Direction Prediction Analysis
# =============================================================================
print("\n[2] Direction Prediction Metrics")

direction_results = []
for model in config.ALL_MODELS:
    aes_col = config.get_rol_avg_col(model, 'simple')
    
    # Actual vs predicted direction
    df_periods['actual_up'] = (df_periods['NRS'] > df_periods['LRS']).astype(int)
    df_periods['pred_up'] = (df_periods[aes_col] > df_periods['LRS']).astype(int)
    
    valid = df_periods[['actual_up', 'pred_up']].dropna()
    
    if len(valid) > 0:
        cm = confusion_matrix(valid['actual_up'], valid['pred_up'])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        direction_results.append({
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_pos': tp,
            'true_neg': tn,
            'false_pos': fp,
            'false_neg': fn,
            'n_obs': len(valid)
        })
        print(f"    {model}: Accuracy={accuracy:.3f}, F1={f1:.3f}")

df_direction = pd.DataFrame(direction_results)
utils.save_table_csv(df_direction, 'enhancements_direction_metrics')

# =============================================================================
# 3. Report Length Effect
# =============================================================================
print("\n[3] Report Length Effect")

if 'report_lengh_verified' in df_periods.columns:
    p95 = df_periods['report_lengh_verified'].quantile(0.95)
    df_periods['report_length_capped'] = df_periods['report_lengh_verified'].clip(upper=p95)
    
    for model in ['base', 'ft']:
        miss_col = config.get_miss_abs_col(model, 'simple')
        valid = df_periods[['report_length_capped', miss_col]].dropna()
        corr = valid['report_length_capped'].corr(valid[miss_col])
        print(f"    {model}: Length-Error r={corr:.4f}")

# =============================================================================
# 4. Temporal Performance
# =============================================================================
print("\n[4] Performance Over Time")

temporal_results = []
for year in sorted(df_periods['year'].unique()):
    df_year = df_periods[df_periods['year'] == year]
    
    year_data = {'year': year, 'n_events': len(df_year)}
    
    for model in config.ALL_MODELS:
        miss_col = config.get_miss_abs_col(model, 'simple')
        mae = df_year[miss_col].mean() if miss_col in df_year.columns else np.nan
        year_data[f'{model}_mae'] = mae
    
    temporal_results.append(year_data)

df_temporal = pd.DataFrame(temporal_results)
utils.save_table_csv(df_temporal, 'enhancements_temporal_analysis')

# Print temporal summary for base model
for _, row in df_temporal.iterrows():
    print(f"    {int(row['year'])}: MAE={row['base_mae']:.2f} (N={int(row['n_events'])})")

print("\n" + "-" * 60)
print("Diagnostic analyses complete")
print("Saved: enhancements_*.csv")
