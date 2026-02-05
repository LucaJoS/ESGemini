"""
HYPOTHESIS 4 ROBUSTNESS: IN-SAMPLE VS OUT-OF-SAMPLE ECONOMIC VALUE
Mirrors the H1 in-sample vs out-of-sample MAE analysis for returns.

Tests whether economic value (alpha) generalizes from in-sample to out-of-sample.
- In-sample: 2017-2023 (training/validation periods)
- Out-of-sample: 2024 (test period)
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils

print("\n" + "="*80)
print("H4 ROBUSTNESS: IN-SAMPLE VS OUT-OF-SAMPLE ECONOMIC VALUE")
print("="*80)

# Load CRSP returns
df_crsp = utils.load_crsp_returns()

# Load data
df_merged = utils.load_working_data('merged_dataset')
df_periods = utils.load_working_data('period_dataset')

# Get ESG release events
events = df_periods[['instrument', 'date', 'NRS', 'LRS', 'supersector']].copy()
events.rename(columns={'date': 'event_date'}, inplace=True)

# Add year column
events['year'] = pd.to_datetime(events['event_date']).dt.year

# Define sample splits
# In-sample: All years before 2024 (training + validation)
# Out-of-sample: 2024 only (test)
events['sample'] = events['year'].apply(lambda x: 'out_of_sample' if x >= 2024 else 'in_sample')

print(f"\nSample distribution:")
print(events['sample'].value_counts())
print(f"\nYear distribution:")
print(events['year'].value_counts().sort_index())

# =============================================================================
# COMPUTE RETURNS BY SAMPLE
# =============================================================================

results_by_sample = []

# Primary window: t-3 formation, t+5 holding (as specified in paper)
FORMATION_DAYS = 3
HOLDING_DAYS = 5

# Focus on base and ft models (mirroring MAE analysis)
FOCUS_MODELS = ['base', 'ft']

for model in FOCUS_MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")
    
    aes_col = config.get_rol_avg_col(model, 'simple')
    
    # Get signals at formation day
    df_signals = df_merged[df_merged['days_to_next_release'] == FORMATION_DAYS].copy()
    df_signals = df_signals[['instrument', 'date', aes_col]].rename(columns={'date': 'signal_date'})
    
    # Merge with events
    df_events = events.merge(df_signals, on='instrument')
    df_events['date_diff'] = (df_events['event_date'] - df_events['signal_date']).dt.days
    df_events = df_events[df_events['date_diff'] == FORMATION_DAYS].copy()
    
    # Calculate signal
    df_events['signal_A'] = df_events[aes_col] - df_events['LRS']
    
    # Process each sample separately
    for sample_type in ['in_sample', 'out_of_sample']:
        print(f"\n  {sample_type.upper()}:")
        
        df_sample = df_events[df_events['sample'] == sample_type].copy()
        
        if len(df_sample) == 0:
            print(f"    No events in {sample_type}")
            continue
        
        returns_list = []
        
        for idx, event_row in df_sample.iterrows():
            inst = event_row['instrument']
            event_date = event_row['event_date']
            
            # Get CRSP return
            ret_cum, date_gap, ticker = utils.get_crsp_cumulative_return(
                df_crsp, inst, event_date, HOLDING_DAYS
            )
            
            if ret_cum is None:
                continue
            
            returns_list.append({
                'instrument': inst,
                'event_date': event_date,
                'signal_A': event_row['signal_A'],
                'return': ret_cum,
                'year': event_row['year']
            })
        
        if len(returns_list) < 10:  # Need minimum events for quintile sorting
            print(f"    Insufficient events: {len(returns_list)}")
            continue
        
        df_strategy = pd.DataFrame(returns_list)
        
        # Quintile sort
        try:
            df_strategy['quintile'] = pd.qcut(
                df_strategy['signal_A'],
                q=config.N_QUINTILES,
                labels=False,
                duplicates='drop'
            ) + 1
        except ValueError as e:
            print(f"    Quintile error: {e}")
            continue
        
        quintile_returns = df_strategy.groupby('quintile')['return'].mean()
        
        if 5 in quintile_returns.index and 1 in quintile_returns.index:
            long_ret = quintile_returns[5]
            short_ret = quintile_returns[1]
            ls_return_gross = long_ret - short_ret
            
            # Transaction costs
            tc = 4 * (config.TRANSACTION_COST_BPS / 10000)
            ls_return_net = ls_return_gross - tc
            
            # T-test
            q5_returns = df_strategy[df_strategy['quintile'] == 5]['return'].values
            q1_returns = df_strategy[df_strategy['quintile'] == 1]['return'].values
            t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns)
            
            results_by_sample.append({
                'model': model,
                'sample': sample_type,
                'n_events': len(df_strategy),
                'long_ret': long_ret,
                'short_ret': short_ret,
                'ls_gross': ls_return_gross,
                'ls_net': ls_return_net,
                't_stat': t_stat,
                'p_val': p_val,
                'formation': f't-{FORMATION_DAYS}',
                'holding': f't+{HOLDING_DAYS}'
            })
            
            print(f"    N Events: {len(df_strategy)}")
            print(f"    Long (Q5): {long_ret*100:+.2f}%")
            print(f"    Short (Q1): {short_ret*100:+.2f}%")
            print(f"    L-S Gross: {ls_return_gross*100:+.2f}%")
            print(f"    L-S Net: {ls_return_net*100:+.2f}% (t={t_stat:.2f}, p={p_val:.3f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================

df_results = pd.DataFrame(results_by_sample)
utils.save_table_csv(df_results, 'h4_in_vs_out_sample_returns')

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("SUMMARY: IN-SAMPLE VS OUT-OF-SAMPLE RETURNS")
print("="*80)
print(f"Window: t-{FORMATION_DAYS} formation / t+{HOLDING_DAYS} holding")
print("-"*80)

if len(df_results) > 0:
    # Pivot for nice display
    pivot = df_results.pivot(index='model', columns='sample', values=['ls_net', 't_stat', 'p_val', 'n_events'])
    print("\nNet Returns (%):")
    for model in FOCUS_MODELS:
        if model in pivot.index:
            in_ret = pivot.loc[model, ('ls_net', 'in_sample')] * 100 if ('ls_net', 'in_sample') in pivot.columns else np.nan
            out_ret = pivot.loc[model, ('ls_net', 'out_of_sample')] * 100 if ('ls_net', 'out_of_sample') in pivot.columns else np.nan
            print(f"  {model:8s}: In-Sample={in_ret:+.2f}%  Out-of-Sample={out_ret:+.2f}%")
    
    print("\nStatistical Significance:")
    for model in FOCUS_MODELS:
        if model in pivot.index:
            in_p = pivot.loc[model, ('p_val', 'in_sample')] if ('p_val', 'in_sample') in pivot.columns else np.nan
            out_p = pivot.loc[model, ('p_val', 'out_of_sample')] if ('p_val', 'out_of_sample') in pivot.columns else np.nan
            in_sig = "**" if in_p < 0.05 else ("*" if in_p < 0.10 else "")
            out_sig = "**" if out_p < 0.05 else ("*" if out_p < 0.10 else "")
            print(f"  {model:8s}: In-Sample p={in_p:.3f}{in_sig}  Out-of-Sample p={out_p:.3f}{out_sig}")
    
    print("\nSample Sizes:")
    for model in FOCUS_MODELS:
        if model in pivot.index:
            in_n = int(pivot.loc[model, ('n_events', 'in_sample')]) if ('n_events', 'in_sample') in pivot.columns else 0
            out_n = int(pivot.loc[model, ('n_events', 'out_of_sample')]) if ('n_events', 'out_of_sample') in pivot.columns else 0
            print(f"  {model:8s}: In-Sample N={in_n}  Out-of-Sample N={out_n}")

print("\n" + "="*80)
print("✓ H4 IN-VS-OUT ROBUSTNESS COMPLETE")
print("="*80)
print(f"Saved: h4_in_vs_out_sample_returns.csv")
print("\nInterpretation:")
print("- If out-of-sample return is similar to in-sample: Signal generalizes well")
print("- If out-of-sample return is weaker: Possible in-sample overfitting")
print("- If out-of-sample return is stronger: 2024 may have more exploitable inefficiency")

