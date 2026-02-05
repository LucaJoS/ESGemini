"""
Script 2: Replication Fidelity (H1)

Purpose:
    Tests whether LLM predictions can replicate Refinitiv ESG scores.
    Computes MAE, RMSE, R2, Spearman correlation, and directional accuracy.

Input:
    - results/data/working_data.db (period_dataset)

Output:
    - results/csv/h1_results.csv
"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config, utils

print("H1: REPLICATION FIDELITY")
df_periods = utils.load_working_data('period_dataset')

# =============================================================================
# CARRY-FORWARD BASELINE (NEW)
# =============================================================================
print("\n--- Carry-Forward Baseline ---")
df_periods['baseline_error'] = df_periods['LRS'] - df_periods['NRS']
df_periods['baseline_error_abs'] = np.abs(df_periods['LRS'] - df_periods['NRS'])
valid_cf = df_periods['LRS'].notna() & df_periods['NRS'].notna()
baseline_mae = df_periods.loc[valid_cf, 'baseline_error_abs'].mean()
baseline_rmse = np.sqrt((df_periods.loc[valid_cf, 'baseline_error'] ** 2).mean())
baseline_r2 = r2_score(df_periods.loc[valid_cf, 'NRS'], df_periods.loc[valid_cf, 'LRS'])
baseline_spearman = stats.spearmanr(df_periods.loc[valid_cf, 'LRS'], df_periods.loc[valid_cf, 'NRS'])[0]
print(f"Carry-Forward: MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}, R²={baseline_r2:.4f}, Spearman={baseline_spearman:.4f}")

# =============================================================================
# MODEL PERFORMANCE (ORIGINAL + NEW METRICS)
# =============================================================================
results = {}
for model in config.ALL_MODELS:
    for roltype in ['simple']:  # Focus on simple for now
        miss_col = config.get_miss_col(model, roltype)
        miss_abs_col = config.get_miss_abs_col(model, roltype)
        
        # Filter valid observations
        valid_mask = df_periods[miss_abs_col].notna()
        df_valid = df_periods[valid_mask]
        
        # MAE (original)
        mae = df_valid[miss_abs_col].mean()
        
        # RMSE (NEW)
        rmse = np.sqrt((df_valid[miss_col] ** 2).mean())
        
        # R² and Spearman (NEW) - reconstruct predictions from miss
        predictions = df_valid['NRS'] + df_valid[miss_col]
        actuals = df_valid['NRS']
        r2 = r2_score(actuals, predictions) if len(df_valid) > 1 else np.nan
        spearman_corr = stats.spearmanr(predictions, actuals)[0] if len(df_valid) > 1 else np.nan
        
        # Compute direction accuracy from prediction errors
        # Direction correct = predicted change sign matches actual change sign
        # predicted_change = actual_change + miss (since miss = prediction - actual)
        df_sorted = df_periods.sort_values(['instrument', 'date']).copy()
        df_sorted['prior_esg'] = df_sorted.groupby('instrument')['esg_score'].shift(1)
        df_sorted['actual_change'] = df_sorted['esg_score'] - df_sorted['prior_esg']
        df_sorted['pred_change'] = df_sorted['actual_change'] + df_sorted[miss_col]
        
        # Valid rows: have both actual change and prediction
        valid_dir = df_sorted['actual_change'].notna() & df_sorted[miss_col].notna()
        if valid_dir.sum() > 0:
            actual = df_sorted.loc[valid_dir, 'actual_change']
            predicted = df_sorted.loc[valid_dir, 'pred_change']
            # Direction correct if signs match (both positive, both negative, or both ~0)
            correct = ((actual > 0) & (predicted > 0)) | ((actual < 0) & (predicted < 0)) | ((actual.abs() < 0.01) & (predicted.abs() < 0.01))
            dir_acc = correct.sum() / len(correct) * 100
        else:
            dir_acc = np.nan
        
        results[f"{model}_{roltype}"] = {
            'MAE': mae, 
            'RMSE': rmse,
            'R2': r2,
            'Spearman': spearman_corr,
            'Dir': dir_acc,
            'N': len(df_valid)
        }
        dir_str = f"{dir_acc:.1f}%" if not np.isnan(dir_acc) else "N/A"
        print(f"{model}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, Spearman={spearman_corr:.4f}, Dir={dir_str} (n={valid_dir.sum()})")

# Add carry-forward baseline to results
results['carry_forward'] = {
    'MAE': baseline_mae,
    'RMSE': baseline_rmse,
    'R2': baseline_r2,
    'Spearman': baseline_spearman,
    'Dir': 0.0,  # Always predicts "no change"
    'N': valid_cf.sum()
}

# =============================================================================
# IN vs OUT SAMPLE (PRESERVED)
# =============================================================================
for sample in ['in_sample', 'out_of_sample']:
    df_s = df_periods[df_periods[sample] == 1]
    print(f"\n{sample}: N={len(df_s)}")
    for model in config.ALL_MODELS:
        miss_abs_col = config.get_miss_abs_col(model, 'simple')
        miss_col = config.get_miss_col(model, 'simple')
        valid = df_s[miss_abs_col].notna()
        if valid.sum() > 0:
            mae = df_s.loc[valid, miss_abs_col].mean()
            rmse = np.sqrt((df_s.loc[valid, miss_col] ** 2).mean())
            print(f"  {model}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# =============================================================================
# ROLLING AVERAGE VARIANTS (NEW)
# =============================================================================
print("\n--- Rolling Average Type Comparison ---")
for model in config.ALL_MODELS:
    print(f"\n{model}:")
    for roltype in ['simple', 'cert', 'length']:
        miss_abs_col = config.get_miss_abs_col(model, roltype)
        valid = df_periods[miss_abs_col].notna()
        if valid.sum() > 0:
            mae = df_periods.loc[valid, miss_abs_col].mean()
            print(f"  {roltype}: MAE={mae:.4f} (n={valid.sum()})")

# Save results
df_results = pd.DataFrame(results).T
utils.save_table_csv(df_results, 'h1_results')
print("\n✓ H1 Complete")
