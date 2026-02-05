"""
Script 5c: FF5-Adjusted Abnormal Returns (H4c)

Purpose:
    Robustness test using Fama-French 5-factor adjusted abnormal returns
    instead of raw returns. Estimates firm-specific factor loadings from
    180-day pre-event window.

Input:
    - results/data/working_data.db
    - data/proprietary___not_available/CRSP_SP100_ReturnData.db
    - data/F-F_Research_Data_5_Factors_2x3_daily.xlsx

Output:
    - results/csv/h4c_ff_adjusted_returns_crsp.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils
from matched_CRSP_ticker import get_crsp_ticker, is_in_data_gap

print("Script 5c: FF5-Adjusted Returns (H4c)")
print("-" * 60)


def estimate_factor_loadings(instrument, end_date, df_crsp, ff_factors, lookback_days=180):
    """Estimate FF5 factor loadings using pre-event historical data."""
    ticker = get_crsp_ticker(instrument, end_date)
    if ticker is None:
        return None
    
    start_date = end_date - pd.Timedelta(days=lookback_days)
    stock_returns = df_crsp[
        (df_crsp['TICKER'] == ticker) &
        (df_crsp['date'] >= start_date) &
        (df_crsp['date'] < end_date)
    ][['date', 'return']].copy()
    
    if len(stock_returns) < 30:
        return None
    
    merged = stock_returns.merge(ff_factors, on='date', how='inner')
    if len(merged) < 30:
        return None
    
    merged['excess_return'] = merged['return'] - (merged['RF'] / 100)
    
    X = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']] / 100
    X = sm.add_constant(X)
    y = merged['excess_return']
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[valid_idx], y[valid_idx]
    
    if len(y) < 20:
        return None
    
    try:
        model = sm.OLS(y, X).fit()
        return {
            'alpha': model.params['const'],
            'beta_mkt': model.params['Mkt-RF'],
            'beta_smb': model.params['SMB'],
            'beta_hml': model.params['HML'],
            'beta_rmw': model.params['RMW'],
            'beta_cma': model.params['CMA'],
            'r_squared': model.rsquared,
            'n_obs': len(y)
        }
    except Exception:
        return None


# Load data
df_crsp = utils.load_crsp_returns()
df_merged = utils.load_working_data('merged_dataset')
df_periods = utils.load_working_data('period_dataset')

print("\nLoading Fama-French factors...")
ff_data = pd.read_excel(config.FF_FACTORS_FILE)
ff_data['date'] = pd.to_datetime(ff_data['date'])
print(f"    Loaded {len(ff_data)} daily observations")

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
            abnormal_returns_list = []

            for idx, event_row in df_events.iterrows():
                inst = event_row['instrument']
                event_date = event_row['event_date']

                loadings = estimate_factor_loadings(inst, event_date, df_crsp, ff_data)
                if loadings is None:
                    continue

                ret_cum, date_gap, ticker = utils.get_crsp_cumulative_return(
                    df_crsp, inst, event_date, holding_days
                )
                if ret_cum is None:
                    if ticker is None:
                        data_gap_skips.add(inst)
                    continue

                # Get FF factors for holding period
                post_dates = df_crsp[
                    (df_crsp['TICKER'] == ticker) & 
                    (df_crsp['date'] >= event_date)
                ].sort_values('date').head(holding_days + 1)['date'].iloc[1:holding_days+1]
                
                ff_holding = ff_data[ff_data['date'].isin(post_dates)]
                if len(ff_holding) < holding_days:
                    continue

                avg_ff = ff_holding[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']].mean()
                expected_return = (
                    loadings['alpha'] * holding_days +
                    loadings['beta_mkt'] * (avg_ff['Mkt-RF'] / 100) +
                    loadings['beta_smb'] * (avg_ff['SMB'] / 100) +
                    loadings['beta_hml'] * (avg_ff['HML'] / 100) +
                    loadings['beta_rmw'] * (avg_ff['RMW'] / 100) +
                    loadings['beta_cma'] * (avg_ff['CMA'] / 100)
                )

                abnormal_returns_list.append({
                    'instrument': inst,
                    'event_date': event_date,
                    'supersector': event_row['supersector'],
                    'signal_A': event_row['signal_A'],
                    'actual_return': ret_cum,
                    'expected_return': expected_return,
                    'abnormal_return': ret_cum - expected_return,
                    'r_squared': loadings['r_squared']
                })

            if len(abnormal_returns_list) == 0:
                continue

            df_strategy = pd.DataFrame(abnormal_returns_list)
            
            try:
                df_strategy['quintile'] = pd.qcut(
                    df_strategy['signal_A'], q=config.N_QUINTILES,
                    labels=False, duplicates='drop'
                ) + 1
            except ValueError:
                continue

            quintile_ar = df_strategy.groupby('quintile')['abnormal_return'].mean()

            if 5 in quintile_ar.index and 1 in quintile_ar.index:
                long_ar = quintile_ar[5]
                short_ar = quintile_ar[1]
                ls_ar_gross = long_ar - short_ar
                tc = 4 * (config.TRANSACTION_COST_BPS / 10000)
                ls_ar_net = ls_ar_gross - tc

                q5_ar = df_strategy[df_strategy['quintile'] == 5]['abnormal_return'].values
                q1_ar = df_strategy[df_strategy['quintile'] == 1]['abnormal_return'].values

                min_len = min(len(q5_ar), len(q1_ar))
                if min_len > 0:
                    t_stat, p_val = stats.ttest_1samp(q5_ar[:min_len] - q1_ar[:min_len], 0)
                else:
                    t_stat, p_val = np.nan, np.nan

                results_all.append({
                    'model': model,
                    'formation': f't-{formation_days}',
                    'holding': f't+{holding_days}',
                    'method': 'ff_adjusted',
                    'n_events': len(df_strategy),
                    'long_ar': long_ar,
                    'short_ar': short_ar,
                    'ls_ar_gross': ls_ar_gross,
                    'ls_ar_net': ls_ar_net,
                    't_stat': t_stat,
                    'p_val': p_val,
                    'avg_r_squared': df_strategy['r_squared'].mean()
                })

                print(f"    t-{formation_days}/t+{holding_days}: AR={ls_ar_net*100:+.2f}% (p={p_val:.3f})")

# Save results
df_h4c = pd.DataFrame(results_all)
utils.save_table_csv(df_h4c, 'h4c_ff_adjusted_returns_crsp')

print("\n" + "-" * 60)
print("H4c complete")
summary = df_h4c.groupby('model')['ls_ar_net'].mean()
for model, ret in summary.items():
    print(f"    {model}: {ret*100:+.2f}% avg abnormal return")
