"""
Script 5: Economic Value (H4)

Purpose:
    Tests whether anticipatory ESG signals generate abnormal returns around 
    Refinitiv rating releases. Forms quintile portfolios based on predicted
    ESG surprise and measures long-short returns.

Input:
    - results/data/working_data.db (merged_dataset, period_dataset)
    - data/proprietary___not_available/CRSP_SP100_ReturnData.db

Output:
    - results/csv/h4_portfolio_returns_comprehensive_crsp.csv

Methodology:
    - Signal A: AES_prediction - LRS (anticipatable surprise)
    - Signal B: (AES_prediction - LRS) / LRS (relative surprise)
    - Signal DIAGNOSTIC: AES_prediction - NRS (uses future info, not tradable)
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils

print("Script 5: Economic Value (H4)")
print("-" * 60)

# Load data
df_crsp = utils.load_crsp_returns()
df_merged = utils.load_working_data('merged_dataset')
df_periods = utils.load_working_data('period_dataset')

# Prepare events
events = df_periods[['instrument', 'date', 'NRS', 'LRS', 'supersector']].copy()
events.rename(columns={'date': 'event_date'}, inplace=True)

# Validate ticker mapping
all_instruments = df_merged['instrument'].unique()
crsp_tickers = set(df_crsp['TICKER'].unique())
missing_count = 0
for inst in all_instruments:
    sample_date = df_merged[df_merged['instrument'] == inst]['date'].iloc[0]
    crsp_ticker = utils.get_crsp_ticker(inst, sample_date)
    if not (crsp_ticker and crsp_ticker in crsp_tickers):
        missing_count += 1

if missing_count > 0:
    print(f"    Warning: {missing_count} instruments missing CRSP mapping")
else:
    print(f"    All {len(all_instruments)} instruments mapped to CRSP")

# Track data quality
data_gap_skips = set()
results_all = []

# Test all models
for model in config.ALL_MODELS:
    print(f"\n[{model}]")
    aes_col = config.get_rol_avg_col(model, 'simple')

    for formation_days in config.FORMATION_PERIODS:
        df_signals = df_merged[df_merged['days_to_next_release'] == formation_days].copy()
        df_signals = df_signals[['instrument', 'date', aes_col]].rename(columns={'date': 'signal_date'})

        df_events = events.merge(df_signals, on='instrument')
        df_events['date_diff'] = (df_events['event_date'] - df_events['signal_date']).dt.days
        df_events = df_events[df_events['date_diff'] == formation_days].copy()

        # Compute signals
        df_events['signal_A'] = df_events[aes_col] - df_events['LRS']
        df_events['signal_B'] = (df_events[aes_col] - df_events['LRS']) / df_events['LRS']
        # Diagnostic signal uses future information (NRS) - NOT tradable
        df_events['signal_DIAGNOSTIC'] = df_events[aes_col] - df_events['NRS']

        for holding_days in config.HOLDING_PERIODS:
            returns_list = []

            for idx, event_row in df_events.iterrows():
                inst = event_row['instrument']
                event_date = event_row['event_date']

                ret_cum, date_gap, ticker = utils.get_crsp_cumulative_return(
                    df_crsp, inst, event_date, holding_days
                )
                
                if ret_cum is None:
                    if ticker is None and inst not in data_gap_skips:
                        data_gap_skips.add(inst)
                    continue

                returns_list.append({
                    'instrument': inst,
                    'event_date': event_date,
                    'supersector': event_row['supersector'],
                    'signal_A': event_row['signal_A'],
                    'signal_B': event_row['signal_B'],
                    'signal_DIAGNOSTIC': event_row['signal_DIAGNOSTIC'],
                    'return': ret_cum,
                    'NRS': event_row['NRS'],
                    'LRS': event_row['LRS']
                })

            if len(returns_list) == 0:
                continue

            df_strategy = pd.DataFrame(returns_list)

            for signal_type in ['signal_A', 'signal_B', 'signal_DIAGNOSTIC']:
                try:
                    df_strategy['quintile'] = pd.qcut(
                        df_strategy[signal_type],
                        q=config.N_QUINTILES,
                        labels=False,
                        duplicates='drop'
                    ) + 1
                except ValueError:
                    continue

                quintile_returns = df_strategy.groupby('quintile')['return'].mean()

                if 5 in quintile_returns.index and 1 in quintile_returns.index:
                    long_ret = quintile_returns[5]
                    short_ret = quintile_returns[1]
                    ls_gross = long_ret - short_ret
                    tc = 4 * (config.TRANSACTION_COST_BPS / 10000)
                    ls_net = ls_gross - tc

                    q5_returns = df_strategy[df_strategy['quintile'] == 5]['return'].values
                    q1_returns = df_strategy[df_strategy['quintile'] == 1]['return'].values
                    t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns)

                    results_all.append({
                        'model': model,
                        'formation': f't-{formation_days}',
                        'holding': f't+{holding_days}',
                        'signal': signal_type,
                        'n_events': len(df_strategy),
                        'long_ret': long_ret,
                        'short_ret': short_ret,
                        'ls_gross': ls_gross,
                        'ls_net': ls_net,
                        't_stat': t_stat,
                        'p_val': p_val
                    })

                    # Print primary signal results
                    if signal_type == 'signal_A':
                        print(f"    t-{formation_days}/t+{holding_days}: L-S={ls_net*100:+.2f}% (p={p_val:.3f})")

# Save results
df_h4 = pd.DataFrame(results_all)
utils.save_table_csv(df_h4, 'h4_portfolio_returns_comprehensive_crsp')

# Summary
print("\n" + "-" * 60)
print("H4 complete")
summary = df_h4[df_h4['signal'] == 'signal_A'].groupby('model')['ls_net'].mean()
for model, ret in summary.items():
    print(f"    {model}: {ret*100:+.2f}% avg net return")
print("\nNote: signal_DIAGNOSTIC uses future info (NRS) - for analysis only")
