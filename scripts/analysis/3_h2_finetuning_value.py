"""
Script 3: Value of Fine-Tuning (H2)

Purpose:
    Tests whether fine-tuning improves prediction accuracy using
    formal statistical tests.

Statistical Tests:
    - Diebold-Mariano (1995): Non-nested model comparison (base vs ft)
    - Clark-West (2007): Nested model comparison (base vs base_a)

Comparisons:
    - Base vs Fine-Tuned (non-nested)
    - Base vs Base+Anchored (nested - anchoring adds parameters)
    - Base_a vs FT_a (non-nested, fair comparison)

Input:
    - results/data/working_data.db (period_dataset)

Output:
    - results/csv/h2_statistical_tests.csv
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config, utils

print("\n" + "="*80)
print("H2: VALUE OF FINE-TUNING - COMPREHENSIVE ANALYSIS")
print("="*80)

df_periods = utils.load_working_data('period_dataset')
print(f"Total periods: {len(df_periods)}")

# =============================================================================
# HELPER: Normal CDF approximation (avoid scipy)
# =============================================================================
def norm_cdf(x):
    """Standard normal CDF using error function approximation."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# =============================================================================
# DIEBOLD-MARIANO TEST (1995)
# =============================================================================
def diebold_mariano_test(errors1, errors2, h=1, loss='squared'):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    H0: E[d_t] = 0 (equal predictive accuracy)
    H1: E[d_t] != 0 (different predictive accuracy)
    
    Parameters:
    - errors1, errors2: Forecast errors from two models
    - h: Forecast horizon (for HAC variance estimation)
    - loss: 'squared' for MSE, 'absolute' for MAE
    
    Returns:
    - DM statistic, p-value
    """
    # Ensure same length and no NaN
    mask = ~(np.isnan(errors1) | np.isnan(errors2))
    e1 = np.array(errors1[mask])
    e2 = np.array(errors2[mask])
    
    if len(e1) < 10:
        return np.nan, np.nan
    
    # Calculate loss differential
    if loss == 'squared':
        d = e1**2 - e2**2
    else:  # absolute
        d = np.abs(e1) - np.abs(e2)
    
    n = len(d)
    d_bar = np.mean(d)
    
    # HAC variance estimation (Newey-West style)
    # For h-step ahead, use h-1 lags
    gamma_0 = np.var(d, ddof=1)
    
    # Autocovariances
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n
        gamma_sum += 2 * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    
    if var_d <= 0:
        return np.nan, np.nan
    
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - norm_cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# =============================================================================
# CLARK-WEST TEST (2007)
# =============================================================================
def clark_west_test(errors_restricted, errors_unrestricted, predictions_restricted, predictions_unrestricted):
    """
    Clark-West test for nested model comparison.
    
    Adjusts for the fact that the unrestricted model estimates more parameters,
    which introduces noise that biases the MSPE comparison.
    
    H0: Restricted model is as good as unrestricted
    H1: Unrestricted model is better (one-sided)
    
    Adjustment: f_{t+1} = e1^2 - [e2^2 - (y1_hat - y2_hat)^2]
    
    Returns:
    - CW statistic, p-value (one-sided)
    """
    # Ensure same length and no NaN
    mask = ~(np.isnan(errors_restricted) | np.isnan(errors_unrestricted) | 
             np.isnan(predictions_restricted) | np.isnan(predictions_unrestricted))
    
    e1 = np.array(errors_restricted[mask])
    e2 = np.array(errors_unrestricted[mask])
    y1_hat = np.array(predictions_restricted[mask])
    y2_hat = np.array(predictions_unrestricted[mask])
    
    if len(e1) < 10:
        return np.nan, np.nan
    
    # Clark-West adjustment
    adjustment = (y1_hat - y2_hat)**2
    f = e1**2 - (e2**2 - adjustment)
    
    n = len(f)
    f_bar = np.mean(f)
    var_f = np.var(f, ddof=1) / n
    
    if var_f <= 0:
        return np.nan, np.nan
    
    cw_stat = f_bar / np.sqrt(var_f)
    p_value = 1 - norm_cdf(cw_stat)  # One-sided
    
    return cw_stat, p_value

# =============================================================================
# MAIN COMPARISONS
# =============================================================================

results = {}

print("\n" + "="*80)
print("1. BASIC MAE COMPARISONS")
print("="*80)

for roltype in ['simple', 'cert', 'length']:
    print(f"\n--- Rolling Average Type: {roltype} ---")
    
    for model in config.ALL_MODELS:
        miss_col = config.get_miss_abs_col(model, roltype)
        mae = df_periods[miss_col].mean()
        n = df_periods[miss_col].notna().sum()
        print(f"  {model}: MAE = {mae:.4f} (n={n})")

# =============================================================================
# DIEBOLD-MARIANO TESTS (Non-nested comparisons)
# =============================================================================
print("\n" + "="*80)
print("2. DIEBOLD-MARIANO TESTS (Non-nested comparisons)")
print("="*80)

# Get errors (signed, not absolute)
base_errors = df_periods[config.get_miss_col('base', 'simple')]
ft_errors = df_periods[config.get_miss_col('ft', 'simple')]
base_a_errors = df_periods[config.get_miss_col('base_a', 'simple')]
ft_a_errors = df_periods[config.get_miss_col('ft_a', 'simple')]

# Test 1: base vs ft (non-nested)
dm_stat, dm_p = diebold_mariano_test(base_errors, ft_errors, h=1, loss='squared')
print(f"\nBase vs FT (MSE):")
print(f"  DM statistic: {dm_stat:.4f}")
print(f"  p-value: {dm_p:.4f}")
print(f"  Interpretation: {'Base significantly better' if dm_stat < 0 and dm_p < 0.05 else 'FT significantly better' if dm_stat > 0 and dm_p < 0.05 else 'No significant difference'}")

results['dm_base_vs_ft'] = {'stat': dm_stat, 'p': dm_p}

# Test 2: base vs ft (MAE loss)
dm_stat_mae, dm_p_mae = diebold_mariano_test(base_errors, ft_errors, h=1, loss='absolute')
print(f"\nBase vs FT (MAE):")
print(f"  DM statistic: {dm_stat_mae:.4f}")
print(f"  p-value: {dm_p_mae:.4f}")

results['dm_base_vs_ft_mae'] = {'stat': dm_stat_mae, 'p': dm_p_mae}

# Test 3: base_a vs ft_a (fair comparison)
dm_stat_a, dm_p_a = diebold_mariano_test(base_a_errors, ft_a_errors, h=1, loss='squared')
print(f"\nBase+Anchored vs FT+Anchored (MSE):")
print(f"  DM statistic: {dm_stat_a:.4f}")
print(f"  p-value: {dm_p_a:.4f}")
print(f"  Interpretation: {'Base_a significantly better' if dm_stat_a < 0 and dm_p_a < 0.05 else 'FT_a significantly better' if dm_stat_a > 0 and dm_p_a < 0.05 else 'No significant difference'}")

results['dm_base_a_vs_ft_a'] = {'stat': dm_stat_a, 'p': dm_p_a}

# =============================================================================
# CLARK-WEST TESTS (Nested comparisons)
# =============================================================================
print("\n" + "="*80)
print("3. CLARK-WEST TESTS (Nested comparisons)")
print("="*80)

# For Clark-West, we need predictions not just errors
# miss = rolling_avg - NRS, so rolling_avg = NRS + miss
nrs = df_periods['NRS']

base_pred = nrs + df_periods[config.get_miss_col('base', 'simple')]
base_a_pred = nrs + df_periods[config.get_miss_col('base_a', 'simple')]
ft_pred = nrs + df_periods[config.get_miss_col('ft', 'simple')]
ft_a_pred = nrs + df_periods[config.get_miss_col('ft_a', 'simple')]

# Test 1: base vs base_a (anchoring adds information = unrestricted)
cw_stat, cw_p = clark_west_test(base_errors, base_a_errors, base_pred, base_a_pred)
print(f"\nBase vs Base+Anchored (Anchoring as unrestricted):")
print(f"  CW statistic: {cw_stat:.4f}")
print(f"  p-value (one-sided): {cw_p:.4f}")
print(f"  Interpretation: {'Anchoring significantly improves prediction' if cw_stat > 1.645 else 'No significant improvement from anchoring'}")

results['cw_base_vs_base_a'] = {'stat': cw_stat, 'p': cw_p}

# Test 2: ft vs ft_a
cw_stat_ft, cw_p_ft = clark_west_test(ft_errors, ft_a_errors, ft_pred, ft_a_pred)
print(f"\nFT vs FT+Anchored (Anchoring as unrestricted):")
print(f"  CW statistic: {cw_stat_ft:.4f}")
print(f"  p-value (one-sided): {cw_p_ft:.4f}")
print(f"  Interpretation: {'Anchoring significantly improves FT prediction' if cw_stat_ft > 1.645 else 'No significant improvement from anchoring'}")

results['cw_ft_vs_ft_a'] = {'stat': cw_stat_ft, 'p': cw_p_ft}

# =============================================================================
# CARRY-FORWARD COMPARISON
# =============================================================================
print("\n" + "="*80)
print("4. COMPARISON TO CARRY-FORWARD BASELINE")
print("="*80)

# Carry-forward error: LRS - NRS (predicting no change)
cf_errors = df_periods['LRS'] - df_periods['NRS']

# DM test: each model vs carry-forward
for model in config.ALL_MODELS:
    model_errors = df_periods[config.get_miss_col(model, 'simple')]
    dm_stat, dm_p = diebold_mariano_test(cf_errors, model_errors, h=1, loss='squared')
    print(f"\nCarry-Forward vs {model}:")
    print(f"  DM statistic: {dm_stat:.4f}, p-value: {dm_p:.4f}")
    
    # Negative DM means first model (CF) is better
    if dm_p < 0.05:
        if dm_stat < 0:
            print(f"  → Carry-Forward significantly better than {model}")
        else:
            print(f"  → {model} significantly better than Carry-Forward")
    else:
        print(f"  → No significant difference")
    
    results[f'dm_cf_vs_{model}'] = {'stat': dm_stat, 'p': dm_p}

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: STATISTICAL TEST RESULTS")
print("="*80)

print("\n| Comparison | Test | Statistic | p-value | Result |")
print("|------------|------|-----------|---------|--------|")
print(f"| Base vs FT | DM | {results['dm_base_vs_ft']['stat']:.3f} | {results['dm_base_vs_ft']['p']:.4f} | {'Base better*' if results['dm_base_vs_ft']['p'] < 0.05 and results['dm_base_vs_ft']['stat'] < 0 else 'FT better*' if results['dm_base_vs_ft']['p'] < 0.05 and results['dm_base_vs_ft']['stat'] > 0 else 'n.s.'} |")
print(f"| Base_a vs FT_a | DM | {results['dm_base_a_vs_ft_a']['stat']:.3f} | {results['dm_base_a_vs_ft_a']['p']:.4f} | {'Base_a better*' if results['dm_base_a_vs_ft_a']['p'] < 0.05 and results['dm_base_a_vs_ft_a']['stat'] < 0 else 'FT_a better*' if results['dm_base_a_vs_ft_a']['p'] < 0.05 and results['dm_base_a_vs_ft_a']['stat'] > 0 else 'n.s.'} |")
print(f"| Base → Base_a | CW | {results['cw_base_vs_base_a']['stat']:.3f} | {results['cw_base_vs_base_a']['p']:.4f} | {'Anchoring helps*' if results['cw_base_vs_base_a']['p'] < 0.05 else 'n.s.'} |")
print(f"| FT → FT_a | CW | {results['cw_ft_vs_ft_a']['stat']:.3f} | {results['cw_ft_vs_ft_a']['p']:.4f} | {'Anchoring helps*' if results['cw_ft_vs_ft_a']['p'] < 0.05 else 'n.s.'} |")

print("\n* = statistically significant at 5% level")
print("DM = Diebold-Mariano (two-sided), CW = Clark-West (one-sided)")

# =============================================================================
# SAVE RESULTS
# =============================================================================
df_results = pd.DataFrame(results).T
utils.save_table_csv(df_results, 'h2_statistical_tests')

print("\n" + "="*80)
print("✓ H2 COMPLETE")
print("="*80)
print("Saved: h2_statistical_tests.csv")
