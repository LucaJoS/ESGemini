"""
Script 1: Data Preparation

Purpose:
    Loads data from gemini_esg_data.db and prepares datasets for analysis.
    Creates working_data.db with merged and period-level datasets.

Input:
    - data/gemini_esg_data.db (main database with predictions and sample flags)
    - data/proprietary___not_available/S&P100_sectors_hard.sqlite
    - data/proprietary___not_available/S&P100_market_cap_hard.sqlite
    - data/F-F_Research_Data_5_Factors_2x3_daily.xlsx

Output:
    - results/data/working_data.db (merged_dataset, period_dataset tables)
    - results/data/merged_dataset.csv
    - results/data/period_dataset.csv

Sample:
    is_primary_event = 1 (N=720 unique rating change events)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import utils

print("Script 1: Data Preparation")
print("-" * 60)

# -----------------------------------------------------------------------------
# Step 1: Load main database
# -----------------------------------------------------------------------------
print("\n[1] Loading main database...")
df_main = utils.load_sqlite_table(config.MAIN_DB, 'master_predictions')

required_flags = ['is_primary_event', 'primary_sample', 'h4_sample', 'is_2024_test', 'is_excluded_instrument']
missing_flags = [f for f in required_flags if f not in df_main.columns]
if missing_flags:
    print(f"ERROR: Missing flags in database: {missing_flags}")
    sys.exit(1)

df_main['date'] = pd.to_datetime(df_main['date'])

# -----------------------------------------------------------------------------
# Step 2: Verify sample counts
# -----------------------------------------------------------------------------
print("\n[2] Verifying sample counts...")
counts = {
    'Total rows': len(df_main),
    'Primary events': (df_main['is_primary_event'] == 1).sum(),
    '2024 test events': ((df_main['is_primary_event'] == 1) & (df_main['is_2024_test'] == 1)).sum(),
}
for label, count in counts.items():
    print(f"    {label}: {count:,}")

expected_n = config.SAMPLE_COUNTS['is_primary_event']
if counts['Primary events'] != expected_n:
    print(f"    WARNING: Expected {expected_n} primary events, got {counts['Primary events']}")

# -----------------------------------------------------------------------------
# Step 3: Add sample indicators
# -----------------------------------------------------------------------------
print("\n[3] Adding sample indicators...")
df_main['in_sample'] = (
    (df_main['date'] >= config.IN_SAMPLE_START) & 
    (df_main['date'] <= config.IN_SAMPLE_END)
).astype(int)

df_main['out_of_sample'] = (
    (df_main['date'] >= config.OUT_OF_SAMPLE_START) & 
    (df_main['date'] <= config.OUT_OF_SAMPLE_END)
).astype(int)

df_main['year'] = df_main['date'].dt.year

# -----------------------------------------------------------------------------
# Step 4: Load external datasets
# -----------------------------------------------------------------------------
print("\n[4] Loading external datasets...")

# Sectors (required for supersector mapping)
df_sectors = utils.load_sqlite_table(config.SECTORS_DB, 'sectors_hard')
df_sectors['date'] = pd.to_datetime(df_sectors['date'])
value_vars = [col for col in df_sectors.columns if col != 'date']
df_sectors_long = pd.melt(df_sectors, id_vars=['date'], value_vars=value_vars,
                          var_name='instrument', value_name='gics_sector')
df_sectors_long['supersector'] = df_sectors_long['gics_sector'].map(config.SUPERSECTOR_MAPPING)
df_sectors_final = df_sectors_long.sort_values(['instrument', 'date']).groupby('instrument').last().reset_index()
df_sectors_final = df_sectors_final[['instrument', 'gics_sector', 'supersector']]

# Market cap (required for robustness analysis)
df_mcap = utils.load_sqlite_table(config.MARKET_CAP_DB, 'market_cap_hard')
df_mcap['date'] = pd.to_datetime(df_mcap['date'])
value_vars = [col for col in df_mcap.columns if col != 'date']
df_mcap_long = pd.melt(df_mcap, id_vars=['date'], value_vars=value_vars,
                       var_name='instrument', value_name='market_cap')
df_mcap_long['log_market_cap'] = np.log(df_mcap_long['market_cap'].replace(0, np.nan))

# Fama-French factors
df_ff = utils.load_excel_sheet(config.FF_FACTORS_FILE, 'F-F_Research_Data_5_Factors_2x3')
df_ff['date'] = pd.to_datetime(df_ff['date'], format='%d.%m.%y')
df_ff = df_ff.rename(columns={'Mkt-RF': 'mkt_rf', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma', 'RF': 'rf'})
for col in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']:
    df_ff[col] = pd.to_numeric(df_ff[col], errors='coerce') / 100

# -----------------------------------------------------------------------------
# Step 5: Merge datasets
# -----------------------------------------------------------------------------
print("\n[5] Merging datasets...")
df_merged = df_main.copy()
df_merged = df_merged.merge(df_sectors_final, on='instrument', how='left')
df_merged = df_merged.merge(df_mcap_long[['date', 'instrument', 'market_cap', 'log_market_cap']], 
                            on=['date', 'instrument'], how='left')
df_merged = df_merged.merge(df_ff, on='date', how='left')
print(f"    Merged dataset: {len(df_merged):,} rows")

# Note: H4 analyses use CRSP returns directly (not merged returns)

# -----------------------------------------------------------------------------
# Step 6: Create period dataset (primary events only)
# -----------------------------------------------------------------------------
print("\n[6] Creating period dataset...")
df_periods = df_merged[df_merged['is_primary_event'] == 1].copy()

# Calculate period length (days since last rating change)
df_periods = df_periods.sort_values(['instrument', 'date'])
df_periods['period_length_days'] = df_periods.groupby('instrument')['date'].diff().dt.days

# Fill first event with days since first document
first_doc_dates = df_merged[df_merged['base_score'].notna()].groupby('instrument')['date'].min()
df_periods['first_doc_date'] = df_periods['instrument'].map(first_doc_dates)
mask_first = df_periods['period_length_days'].isna()
df_periods.loc[mask_first, 'period_length_days'] = (
    df_periods.loc[mask_first, 'date'] - df_periods.loc[mask_first, 'first_doc_date']
).dt.days
df_periods = df_periods.drop(columns=['first_doc_date'])

# Fill missing supersector
if df_periods['supersector'].isna().any():
    df_periods['supersector'] = df_periods['supersector'].fillna('Other')

print(f"    Period dataset: {len(df_periods):,} events")

# -----------------------------------------------------------------------------
# Step 7: Save datasets
# -----------------------------------------------------------------------------
print("\n[7] Saving datasets...")

# SQLite database
conn = sqlite3.connect(config.WORKING_DATA_DB)
df_merged.to_sql('merged_dataset', conn, if_exists='replace', index=False)
df_periods.to_sql('period_dataset', conn, if_exists='replace', index=False)
metadata = pd.DataFrame({
    'key': ['created_date', 'n_merged', 'n_periods'],
    'value': [pd.Timestamp.now().isoformat(), str(len(df_merged)), str(len(df_periods))]
})
metadata.to_sql('metadata', conn, if_exists='replace', index=False)
conn.close()

# CSV for reference
df_merged.to_csv(os.path.join(config.DATA_DIR, 'merged_dataset.csv'), index=False)
df_periods.to_csv(os.path.join(config.DATA_DIR, 'period_dataset.csv'), index=False)

print(f"    Saved to working_data.db")
print(f"    Saved CSV files")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Data preparation complete")
print(f"  Primary events: {len(df_periods):,}")
print(f"  In-sample (2014-2023): {(df_periods['in_sample'] == 1).sum():,}")
print(f"  Out-of-sample (2024): {(df_periods['out_of_sample'] == 1).sum():,}")
