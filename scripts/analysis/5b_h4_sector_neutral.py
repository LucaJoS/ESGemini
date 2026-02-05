"""
Script 5b: Sector-Neutral Portfolio Strategy (H4b)

Purpose:
    Robustness test forming quintiles WITHIN each supersector,
    controlling for sector-level return patterns.

Input:
    - results/data/working_data.db
    - data/proprietary___not_available/CRSP_SP100_ReturnData.db

Output:
    - results/csv/h4b_sector_neutral_returns_crsp.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils

print("Script 5b: Sector-Neutral Portfolio (H4b)")
print("-" * 60)

# Load data
df_crsp = utils.load_crsp_returns()
df_merged = utils.load_working_data('merged_dataset')
df_periods = utils.load_working_data('period_dataset')

events = df_periods[['instrument', 'date', 'NRS', 'LRS', 'supersector']].copy()
events.rename(columns={'date': 'event_date'}, inplace=True)

data_gap_skips = set()
results_all = []

for model in config.ALL_MODELS:
    print(f"\n[{model}]")
    aes_col = config.get_rol_avg_col(model, 'simple')

    for formation_days in config.FORMATION_PERIODS:
        df_signals = df_merged[df_merged['days_to_next_release'] == formation_days].copy()
        df_signals = df_signals[['instrument', 'date', aes_col]].rename(columns={'date': 'signal_date'})

        df_events = events.merge(df_signals, on='instrument')
        df_events['date_diff'] = (df_events['event_date'] - df_events['signal_date']).dt.days
        df_events = df_events[df_events['date_diff'] == formation_days].copy()
        df_events['signal_A'] = df_events[aes_col] - df_events['LRS']

        for holding_days in config.HOLDING_PERIODS:
            returns_list = []

            for idx, event_row in df_events.iterrows():
                inst = event_row['instrument']
                event_date = event_row['event_date']

                ret_cum, date_gap, ticker = utils.get_crsp_cumulative_return(
                    df_crsp, inst, event_date, holding_days
                )
                
                if ret_cum is None:
                    if ticker is None:
                        data_gap_skips.add(inst)
                    continue

                returns_list.append({
                    'instrument': inst,
                    'event_date': event_date,
                    'supersector': event_row['supersector'],
                    'signal_A': event_row['signal_A'],
                    'return': ret_cum
                })

            if len(returns_list) == 0:
                continue

            df_strategy = pd.DataFrame(returns_list)

            # Sector-neutral: quintiles WITHIN each sector
            df_strategy['quintile'] = df_strategy.groupby('supersector')['signal_A'].transform(
                lambda x: pd.qcut(x, q=config.N_QUINTILES, labels=False, duplicates='drop') + 1
            )

            quintile_returns = df_strategy.groupby('quintile')['return'].mean()

            if 5 in quintile_returns.index and 1 in quintile_returns.index:
                long_ret = quintile_returns[5]
                short_ret = quintile_returns[1]
                ls_gross = long_ret - short_ret
                tc = 4 * (config.TRANSACTION_COST_BPS / 10000)
                ls_net = ls_gross - tc

                q5_returns = df_strategy[df_strategy['quintile'] == 5]['return'].values
                q1_returns = df_strategy[df_strategy['quintile'] == 1]['return'].values

                min_len = min(len(q5_returns), len(q1_returns))
                if min_len > 0:
                    ls_returns = q5_returns[:min_len] - q1_returns[:min_len]
                    t_stat, p_val = stats.ttest_1samp(ls_returns, 0)
                else:
                    t_stat, p_val = np.nan, np.nan

                results_all.append({
                    'model': model,
                    'formation': f't-{formation_days}',
                    'holding': f't+{holding_days}',
                    'method': 'sector_neutral',
                    'n_events': len(df_strategy),
                    'long_ret': long_ret,
                    'short_ret': short_ret,
                    'ls_gross': ls_gross,
                    'ls_net': ls_net,
                    't_stat': t_stat,
                    'p_val': p_val
                })

                print(f"    t-{formation_days}/t+{holding_days}: L-S={ls_net*100:+.2f}% (p={p_val:.3f})")

# Save results
df_h4b = pd.DataFrame(results_all)
utils.save_table_csv(df_h4b, 'h4b_sector_neutral_returns_crsp')

print("\n" + "-" * 60)
print("H4b complete")
summary = df_h4b.groupby('model')['ls_net'].mean()
for model, ret in summary.items():
    print(f"    {model}: {ret*100:+.2f}% avg")
