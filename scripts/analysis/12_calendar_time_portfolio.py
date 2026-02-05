"""
CALENDAR-TIME PORTFOLIO ANALYSIS
"""

import pandas as pd
import numpy as np
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from matched_CRSP_ticker import get_crsp_ticker

print("\n" + "="*80)
print("CALENDAR-TIME PORTFOLIO ANALYSIS")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
FORMATION_DAYS = 3  # Days before event to form portfolio
HOLDING_DAYS = 5    # Days after event to hold
TRANSACTION_COST_BPS = 40

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Load directly from main database (not period_dataset.pkl which has limited coverage)
conn_main = sqlite3.connect(config.MAIN_DB)

# Get all data for signal propagation
df = pd.read_sql_query("""
    SELECT date, instrument, days_to_next_release, 
           base_rol_avg, ft_rol_avg, base_con_rol_avg, ft_con_rol_avg,
           LRS, NRS, esg_score
    FROM master_predictions
    ORDER BY instrument, date
""", conn_main)
conn_main.close()

df['date'] = pd.to_datetime(df['date'])
print(f"  Full dataset: {len(df):,} rows")

# Load CRSP returns
conn_crsp = sqlite3.connect(config.CRSP_DB_PATH)
df_crsp = pd.read_sql_query("SELECT * FROM crsp_sp100_returns", conn_crsp)
conn_crsp.close()
df_crsp['date'] = pd.to_datetime(df_crsp['date'])
df_crsp['RET'] = pd.to_numeric(df_crsp['RET'], errors='coerce')
print(f"  CRSP returns: {len(df_crsp):,} rows")

# Load Fama-French factors
ff_path = config.FF_FACTORS_FILE
if os.path.exists(ff_path):
    df_ff = pd.read_excel(ff_path)
    # Convert date (file already has proper column names)
    df_ff['date'] = pd.to_datetime(df_ff['date'])
    df_ff = df_ff.dropna(subset=['date'])
    # Convert returns to decimal (they're in percentage)
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df_ff[col] = df_ff[col] / 100
    print(f"  Fama-French factors: {len(df_ff):,} rows")
else:
    print(f"  WARNING: FF factors file not found at {ff_path}")
    df_ff = None

# =============================================================================
# IDENTIFY EVENT WINDOWS AND GET LAST SIGNALS
# =============================================================================
print("\n[2] Identifying event windows and computing signals...")

# Rating change events (days_to_next_release = 0)
df_rc = df[df['days_to_next_release'] == 0].copy()
df_rc = df_rc[df_rc['LRS'].notna() & df_rc['NRS'].notna()]
print(f"  Rating change events with LRS/NRS: {len(df_rc):,}")

# For each rating change, get the LAST available signal from the day(s) before
def get_last_signal_before_event(df_full, df_events, signal_col):
    """Get the last available signal value before each rating change event."""
    signals = []
    for idx, event in df_events.iterrows():
        inst = event['instrument']
        event_date = event['date']
        
        # Get all rows for this instrument before the event date
        prior_data = df_full[(df_full['instrument'] == inst) & 
                              (df_full['date'] < event_date) &
                              (df_full[signal_col].notna())]
        
        if len(prior_data) > 0:
            # Get the last (most recent) signal
            last_signal = prior_data.sort_values('date').iloc[-1][signal_col]
            signals.append(last_signal)
        else:
            signals.append(None)
    
    return signals

# Build events DataFrame with signals
df_events = df_rc[['date', 'instrument', 'LRS', 'NRS']].copy()

print("  Computing last available signals before each event...")
for model, col in [('base', 'base_rol_avg'), ('ft', 'ft_rol_avg'), 
                   ('base_a', 'base_con_rol_avg'), ('ft_a', 'ft_con_rol_avg')]:
    df_events[col] = get_last_signal_before_event(df, df_rc, col)
    # Compute anticipation gap (signal - LRS)
    df_events[f'signal_{model}'] = df_events[col].astype(float) - df_events['LRS'].astype(float)
    valid_count = df_events[f'signal_{model}'].notna().sum()
    print(f"    {model}: {valid_count} events with valid signals ({valid_count/len(df_events)*100:.1f}%)")

print(f"  Total events with at least one signal: {len(df_events):,}")

# =============================================================================
# BUILD CALENDAR-TIME PORTFOLIO
# =============================================================================
print("\n[3] Building calendar-time portfolio...")

def build_calendar_time_portfolio(df_events, signal_col, formation_days=3, holding_days=5):
    """
    Build calendar-time portfolio returns.
    For each calendar day, compute the average return of stocks in active event windows.
    """
    # Get all unique calendar dates from CRSP
    all_dates = sorted(df_crsp['date'].unique())
    
    # For each event, determine the active window
    events_with_windows = []
    for idx, row in df_events.iterrows():
        event_date = row['date']
        signal = row[signal_col]
        instrument = row['instrument']
        
        if pd.isna(signal):
            continue
        
        # Formation starts formation_days before event
        # Holding ends holding_days after event
        # Find actual trading dates
        event_idx = np.searchsorted(all_dates, event_date)
        
        # Formation start: Go back formation_days trading days
        form_start_idx = max(0, event_idx - formation_days)
        form_start = all_dates[form_start_idx]
        
        # Holding end: Go forward holding_days trading days
        hold_end_idx = min(len(all_dates) - 1, event_idx + holding_days)
        hold_end = all_dates[hold_end_idx]
        
        events_with_windows.append({
            'instrument': instrument,
            'event_date': event_date,
            'signal': signal,
            'window_start': form_start,
            'window_end': hold_end
        })
    
    df_windows = pd.DataFrame(events_with_windows)
    
    # Sort events into quintiles by signal
    df_windows['quintile'] = pd.qcut(df_windows['signal'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # For each calendar date, compute portfolio return
    portfolio_returns = []
    
    for cal_date in all_dates:
        # Find events where this date is within the active window
        active_long = df_windows[(df_windows['window_start'] <= cal_date) & 
                                  (df_windows['window_end'] >= cal_date) & 
                                  (df_windows['quintile'] == 5)]
        active_short = df_windows[(df_windows['window_start'] <= cal_date) & 
                                   (df_windows['window_end'] >= cal_date) & 
                                   (df_windows['quintile'] == 1)]
        
        if len(active_long) == 0 and len(active_short) == 0:
            continue
        
        # Get returns for active positions
        long_returns = []
        for _, row in active_long.iterrows():
            ticker = get_crsp_ticker(row['instrument'], cal_date.strftime('%Y-%m-%d'))
            if ticker:
                ret_data = df_crsp[(df_crsp['TICKER'] == ticker) & (df_crsp['date'] == cal_date)]
                if not ret_data.empty and not pd.isna(ret_data['RET'].values[0]):
                    long_returns.append(ret_data['RET'].values[0])
        
        short_returns = []
        for _, row in active_short.iterrows():
            ticker = get_crsp_ticker(row['instrument'], cal_date.strftime('%Y-%m-%d'))
            if ticker:
                ret_data = df_crsp[(df_crsp['TICKER'] == ticker) & (df_crsp['date'] == cal_date)]
                if not ret_data.empty and not pd.isna(ret_data['RET'].values[0]):
                    short_returns.append(ret_data['RET'].values[0])
        
        # Equal-weighted portfolio returns
        long_ret = np.mean(long_returns) if long_returns else 0
        short_ret = np.mean(short_returns) if short_returns else 0
        ls_ret = long_ret - short_ret
        
        portfolio_returns.append({
            'date': cal_date,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'ls_ret': ls_ret,
            'n_long': len(long_returns),
            'n_short': len(short_returns)
        })
    
    return pd.DataFrame(portfolio_returns)


def run_ff5_regression(df_portfolio, df_ff):
    """
    Run Fama-French 5-factor regression on calendar-time portfolio returns.
    Returns alpha with HAC-robust standard errors.
    """
    # Merge with FF factors
    df_merged = df_portfolio.merge(df_ff, on='date', how='inner')
    
    if len(df_merged) < 30:
        return None
    
    # Dependent variable: LS return - RF
    df_merged['excess_ret'] = df_merged['ls_ret'] - df_merged['RF']
    
    # Independent variables
    X = df_merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    X = sm.add_constant(X)
    y = df_merged['excess_ret']
    
    # OLS with HAC standard errors (Newey-West)
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    return {
        'alpha': model.params['const'] * 252,  # Annualized (approximately)
        'alpha_daily': model.params['const'],
        't_stat': model.tvalues['const'],
        'p_value': model.pvalues['const'],
        'r_squared': model.rsquared,
        'n_obs': len(df_merged),
        'mkt_beta': model.params['Mkt-RF'],
        'smb_beta': model.params['SMB'],
        'hml_beta': model.params['HML']
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
print("\n[4] Running calendar-time FF5 regressions...")

results = []

for model in ['base', 'base_a', 'ft', 'ft_a']:
    signal_col = f'signal_{model}'
    
    if signal_col not in df_events.columns:
        print(f"  Skipping {model}: signal column not found")
        continue
    
    # Skip if too many NaN values in signal
    if df_events[signal_col].isna().sum() > len(df_events) * 0.8:
        print(f"  Skipping {model}: too many missing values")
        continue
    
    print(f"\n  Processing {model}...")
    
    # Build calendar-time portfolio
    try:
        df_portfolio = build_calendar_time_portfolio(
            df_events, signal_col, FORMATION_DAYS, HOLDING_DAYS
        )
    except Exception as e:
        print(f"    Error building portfolio: {e}")
        continue
    
    if len(df_portfolio) < 30:
        print(f"    Insufficient data: {len(df_portfolio)} days")
        continue
    
    print(f"    Portfolio days: {len(df_portfolio)}")
    print(f"    Avg LS return: {df_portfolio['ls_ret'].mean()*100:.3f}% daily")
    
    # Run FF5 regression
    if df_ff is not None:
        ff_result = run_ff5_regression(df_portfolio, df_ff)
        
        if ff_result:
            # Transaction cost adjustment (rough)
            # Assume turnover based on holding period
            daily_tc = (TRANSACTION_COST_BPS / 10000) / HOLDING_DAYS
            alpha_net = ff_result['alpha_daily'] * 252 - (daily_tc * 252)
            
            results.append({
                'Model': model,
                'Formation': f't-{FORMATION_DAYS}',
                'Holding': f't+{HOLDING_DAYS}',
                'N_Days': ff_result['n_obs'],
                'Alpha_Annual_%': ff_result['alpha'] * 100,
                'Alpha_Net_%': alpha_net * 100,
                't_stat': ff_result['t_stat'],
                'p_value': ff_result['p_value'],
                'R_squared': ff_result['r_squared'],
                'Mkt_Beta': ff_result['mkt_beta'],
                'SMB_Beta': ff_result['smb_beta'],
                'HML_Beta': ff_result['hml_beta'],
                'Significant_5%': ff_result['p_value'] < 0.05,
                'Significant_10%': ff_result['p_value'] < 0.10
            })
            
            print(f"    Alpha (annual): {ff_result['alpha']*100:+.2f}%")
            print(f"    t-stat (HAC): {ff_result['t_stat']:.2f}")
            print(f"    p-value: {ff_result['p_value']:.4f}")
        else:
            print(f"    FF regression failed")
    else:
        print(f"    Skipping FF regression: no factors data")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n[5] Saving results...")

if results:
    df_results = pd.DataFrame(results)
    output_path = os.path.join(config.TABLES_DIR, 'calendar_time_ff5_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # =============================================================================
    # LATEX TABLE
    # =============================================================================
    print("\n[6] Creating LaTeX table...")
    
    latex_table = r"""\begin{table}[htbp]
\centering
\caption{Calendar-Time Portfolio: Fama-French 5-Factor Alpha}
\label{tab:calendartime}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\textbf{Model} & \textbf{Alpha (\%)} & \textbf{t-stat} & \textbf{p-value} & $\bm{R^2}$ & \textbf{MKT $\beta$} & \textbf{N Days} \\
\midrule
"""
    
    for _, row in df_results.iterrows():
        sig = "*" if row['Significant_10%'] else ""
        sig = "**" if row['Significant_5%'] else sig
        latex_table += f"{row['Model']} & {row['Alpha_Annual_%']:+.2f}{sig} & {row['t_stat']:.2f} & {row['p_value']:.3f} & {row['R_squared']:.3f} & {row['Mkt_Beta']:.2f} & {row['N_Days']} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}

\begin{flushleft}
\footnotesize
\textit{Note.} Calendar-time portfolio approach with Fama-French 5-factor adjustment. Alpha is annualized (daily $\times$ 252). T-statistics use HAC (Newey-West) standard errors with 5 lags. Portfolio formed on signal quintiles (Q5 long, Q1 short) with t$-$3 formation and t+5 holding periods. * $p < .10$, ** $p < .05$.
\end{flushleft}
\end{table}
"""
    
    latex_path = os.path.join(config.LATEX_DIR, 'table_calendar_time_ff5.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"  Saved: {latex_path}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("CALENDAR-TIME PORTFOLIO ANALYSIS COMPLETE")
print("="*80)

if results:
    print("\nResults Summary:")
    for _, row in df_results.iterrows():
        sig_status = "✓ Significant" if row['Significant_10%'] else "Not significant"
        print(f"  {row['Model']}: Alpha = {row['Alpha_Annual_%']:+.2f}%, t = {row['t_stat']:.2f}, p = {row['p_value']:.3f} ({sig_status})")
    
    # Compare with event-time results
    print("\nComparison with Event-Time Results:")
    print("  Calendar-time approach handles overlapping events and uses")
    print("  HAC-robust standard errors for more conservative inference.")
    
    ft_row = df_results[df_results['Model'] == 'ft']
    if not ft_row.empty:
        ft_result = ft_row.iloc[0]
        print(f"\n  Fine-tuned model (primary):")
        print(f"    Calendar-time alpha: {ft_result['Alpha_Annual_%']:+.2f}% (p = {ft_result['p_value']:.3f})")
else:
    print("\nNo results generated - check data availability")
