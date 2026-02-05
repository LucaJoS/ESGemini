"""
Utility functions for ESG Prediction Analysis
Shared functions used across multiple analysis scripts
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
import os
warnings.filterwarnings('ignore')

import config

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sqlite_table(db_path, table_name):
    """Load a table from SQLite database"""
    print(f"  Loading {table_name} from {os.path.basename(db_path)}...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    print(f"    ✓ Loaded {len(df):,} rows")
    return df

def load_excel_sheet(file_path, sheet_name):
    """Load sheet from Excel file"""
    print(f"  Loading {sheet_name} from {os.path.basename(file_path)}...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"    ✓ Loaded {len(df):,} rows")
    return df

def load_working_data(table_name):
    """
    Load a table from working_data.db (intermediate analysis data).
    
    Args:
        table_name: str - 'period_dataset' or 'merged_dataset'
    
    Returns:
        DataFrame with the requested data
    """
    db_path = config.WORKING_DATA_DB
    if not os.path.exists(db_path):
        # Fallback to pickle if DB doesn't exist yet
        pkl_path = os.path.join(config.DATA_DIR, f'{table_name}.pkl')
        if os.path.exists(pkl_path):
            print(f"  Loading {table_name} from pickle (DB not found)...")
            return pd.read_pickle(pkl_path)
        raise FileNotFoundError(f"Neither {db_path} nor {pkl_path} found. Run 1_data_preparation.py first.")
    
    print(f"  Loading {table_name} from working_data.db...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    
    # Convert date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
    
    print(f"    ✓ Loaded {len(df):,} rows")
    return df

# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def check_missing_data(df, dataset_name):
    """Check for missing data and report"""
    print(f"\n  Checking {dataset_name} for missing data...")

    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)

    critical_missing = missing_summary[missing_summary > 0]

    if len(critical_missing) == 0:
        print(f"    ✓ No missing data found")
        return True
    else:
        print(f"    ⚠ WARNING: Found missing data:")
        for col, count in critical_missing.items():
            pct = missing_pct[col]
            print(f"      - {col}: {count:,} rows ({pct}%)")

        if config.STOP_ON_MISSING_DATA:
            raise ValueError(f"Missing data found in {dataset_name}. Please fix before proceeding.")
        return False

def validate_date_range(df, date_col, expected_start, expected_end, dataset_name):
    """Validate date range of dataset"""
    df[date_col] = pd.to_datetime(df[date_col])
    actual_start = df[date_col].min()
    actual_end = df[date_col].max()

    print(f"  {dataset_name} date range: {actual_start.date()} to {actual_end.date()}")

    return actual_start, actual_end

# =============================================================================
# SAMPLE SPLITTING FUNCTIONS
# =============================================================================

def add_sample_indicator(df, date_col='date'):
    """Add in_sample and out_of_sample indicators"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df['in_sample'] = (
        (df[date_col] >= config.IN_SAMPLE_START) & 
        (df[date_col] <= config.IN_SAMPLE_END)
    ).astype(int)

    df['out_of_sample'] = (
        (df[date_col] >= config.OUT_OF_SAMPLE_START) & 
        (df[date_col] <= config.OUT_OF_SAMPLE_END)
    ).astype(int)

    df['year'] = df[date_col].dt.year

    return df

def filter_fair_comparison_sample(df, include_ft_c=True):
    """
    Filter DataFrame to fair comparison sample.
    
    Due to inference logic differences, ft_c has fewer observations than other models.
    When comparing ALL 4 models, use fair_comparison_sample = 1 filter.
    When comparing only base/base_c/ft, use full sample.
    
    Args:
        df: DataFrame with fair_comparison_sample column
        include_ft_c: If True, filter to rows where ALL models have data (755 obs)
                      If False, use full sample for base/base_c/ft (949 obs)
    
    Returns:
        Filtered DataFrame
    
    See Post_Inference_Data_Analysis.md for detailed explanation.
    """
    if include_ft_c and config.USE_FAIR_COMPARISON_SAMPLE:
        if 'fair_comparison_sample' in df.columns:
            filtered = df[df['fair_comparison_sample'] == 1].copy()
            print(f"  → Fair comparison filter: {len(df):,} → {len(filtered):,} rows")
            return filtered
        else:
            print("  ⚠ WARNING: fair_comparison_sample column not found, using full sample")
    return df.copy()

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def calculate_mae(errors):
    """Calculate Mean Absolute Error"""
    return np.abs(errors).mean()

def calculate_rmse(errors):
    """Calculate Root Mean Squared Error"""
    return np.sqrt((errors ** 2).mean())

def calculate_direction_accuracy(dir_series):
    """Calculate direction accuracy (% correct predictions)"""
    return (dir_series == 'correct').mean() * 100

def paired_t_test(errors1, errors2):
    """Perform paired t-test between two error series"""
    # Remove NaN pairs
    mask = ~(pd.isna(errors1) | pd.isna(errors2))
    e1 = errors1[mask]
    e2 = errors2[mask]

    if len(e1) < 2:
        return {'t_stat': np.nan, 'p_value': np.nan, 'significant': False}

    t_stat, p_value = stats.ttest_rel(np.abs(e1), np.abs(e2))

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'sig_1pct': p_value < 0.01,
        'sig_5pct': p_value < 0.05
    }

def diebold_mariano_test(errors1, errors2, h=1):
    """Diebold-Mariano test for forecast comparison"""
    # Remove NaN pairs
    mask = ~(pd.isna(errors1) | pd.isna(errors2))
    e1 = np.array(errors1[mask])
    e2 = np.array(errors2[mask])

    if len(e1) < 2:
        return {'dm_stat': np.nan, 'p_value': np.nan}

    # Loss differential (using squared errors)
    d = e1**2 - e2**2

    # Mean loss differential
    d_bar = d.mean()

    # Variance of loss differential
    gamma_0 = np.var(d, ddof=1)

    # DM statistic
    dm_stat = d_bar / np.sqrt(gamma_0 / len(d))

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        'dm_stat': dm_stat,
        'p_value': p_value,
        'sig_1pct': p_value < 0.01,
        'sig_5pct': p_value < 0.05
    }

def mcnemar_test(correct1, correct2):
    """McNemar's test for comparing direction accuracy between two models"""
    from statsmodels.stats.contingency_tables import mcnemar as mc_test

    # Create contingency table
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)

    if b + c == 0:
        return {'chi2': np.nan, 'p_value': np.nan}

    # McNemar's test
    table = [[0, b], [c, 0]]
    result = mc_test(table, exact=False)

    return {
        'chi2': result.statistic,
        'p_value': result.pvalue,
        'sig_1pct': result.pvalue < 0.01,
        'sig_5pct': result.pvalue < 0.05
    }

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std

def apply_bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple testing"""
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    return p_values < adjusted_alpha, adjusted_alpha

def apply_benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction"""
    p_array = np.array(p_values)
    n = len(p_array)

    # Sort p-values
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # Calculate critical values
    critical_values = (np.arange(1, n+1) / n) * alpha

    # Find largest i where p_i <= (i/n)*alpha
    rejections = sorted_p <= critical_values

    if np.any(rejections):
        max_idx = np.where(rejections)[0].max()
        # Reject all hypotheses up to max_idx
        reject_array = np.zeros(n, dtype=bool)
        reject_array[sorted_indices[:max_idx+1]] = True
    else:
        reject_array = np.zeros(n, dtype=bool)

    return reject_array

# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_ols_regression(y, X, cluster_var=None):
    """Run OLS regression with optional clustering"""
    # Add constant
    X_with_const = add_constant(X)

    # Run OLS
    model = OLS(y, X_with_const)

    if cluster_var is not None:
        # Clustered standard errors (firm-level)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_var})
    else:
        results = model.fit()

    return {
        'model': results,
        'coef': results.params,
        'se': results.bse,
        't_stat': results.tvalues,
        'p_value': results.pvalues,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'n_obs': int(results.nobs)
    }

# =============================================================================
# PORTFOLIO FUNCTIONS
# =============================================================================

def create_quintile_portfolios(signals, returns, n_quintiles=5):
    """Create quintile portfolios based on signals"""
    # Remove NaN
    valid_mask = ~(pd.isna(signals) | pd.isna(returns))
    clean_signals = signals[valid_mask]
    clean_returns = returns[valid_mask]

    if len(clean_signals) == 0:
        return {i: np.nan for i in range(1, n_quintiles+1)}

    # Create quintiles
    quintiles = pd.qcut(clean_signals, q=n_quintiles, labels=False, duplicates='drop') + 1

    # Calculate equal-weighted returns for each quintile
    portfolio_returns = {}
    for q in range(1, n_quintiles+1):
        mask = quintiles == q
        if mask.sum() > 0:
            portfolio_returns[q] = clean_returns[mask].mean()
        else:
            portfolio_returns[q] = np.nan

    return portfolio_returns

def calculate_long_short_return(long_return, short_return, transaction_cost_bps=10):
    """Calculate long-short portfolio return with transaction costs"""
    # Convert bps to decimal
    tc = transaction_cost_bps / 10000

    # Long-short return before costs
    gross_return = long_return - short_return

    # Total transaction costs: 
    # Open long (tc) + Close long (tc) + Open short (tc) + Close short (tc) = 4 * tc
    total_tc = 4 * tc

    # Net return after costs
    net_return = gross_return - total_tc

    return net_return

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_table_latex(df, filename, caption="", label=""):
    """Save DataFrame as LaTeX table"""
    filepath = os.path.join(config.LATEX_DIR, f"{filename}.tex")

    latex_str = df.to_latex(
        index=True,
        float_format="%.4f",
        caption=caption,
        label=f"tab:{label}",
        escape=False
    )

    with open(filepath, 'w') as f:
        f.write(latex_str)

    print(f"    ✓ Saved LaTeX: {filename}.tex")

def save_table_csv(df, filename):
    """Save DataFrame as CSV"""
    filepath = os.path.join(config.TABLES_DIR, f"{filename}.csv")
    df.to_csv(filepath, index=True)
    print(f"    ✓ Saved CSV: {filename}.csv")

def save_figure(fig, filename):
    """Save figure as PNG"""
    filepath = os.path.join(config.FIGURES_DIR, f"{filename}.{config.FIG_FORMAT}")
    fig.savefig(filepath, dpi=config.FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved figure: {filename}.{config.FIG_FORMAT}")

def format_pvalue_stars(pvalue):
    """Format p-value with significance stars"""
    if pd.isna(pvalue):
        return ''
    elif pvalue < 0.01:
        return '***'
    elif pvalue < 0.05:
        return '**'
    elif pvalue < 0.10:
        return '*'
    else:
        return ''

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def set_plot_style():
    """Set consistent plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = config.FIG_SIZE_STANDARD
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

def plot_error_distribution(errors_dict, title, xlabel='Error'):
    """Plot error distribution for multiple models"""
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_STANDARD)

    data_to_plot = [errors for errors in errors_dict.values()]
    labels = list(errors_dict.keys())

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Color boxes
    for patch, label in zip(bp['boxes'], labels):
        if label in config.COLOR_MODELS:
            patch.set_facecolor(config.COLOR_MODELS.get(label, '#1f77b4'))

    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel(xlabel)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

def plot_time_series(data_dict, title, ylabel, xlabel='Year'):
    """Plot time series for multiple series"""
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)

    for label, data in data_dict.items():
        color = config.COLOR_MODELS.get(label, None)
        ax.plot(data.index, data.values, marker='o', label=label, color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def plot_bar_comparison(data_dict, title, ylabel):
    """Plot bar chart for model comparison"""
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_STANDARD)

    labels = list(data_dict.keys())
    values = list(data_dict.values())
    colors = [config.COLOR_MODELS.get(label, '#1f77b4') for label in labels]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return fig


# =============================================================================
# CRSP RETURNS FUNCTIONS
# =============================================================================
# Functions for loading and using CRSP returns with proper ticker mapping

# Import CRSP ticker mapping functions
from matched_CRSP_ticker import get_crsp_ticker, is_in_data_gap

def load_crsp_returns():
    """
    Load CRSP returns from the database into memory.
    Returns a DataFrame with TICKER, date, return columns.
    """
    print("Loading CRSP returns from database...")
    conn = sqlite3.connect(config.CRSP_DB_PATH)
    
    df_crsp = pd.read_sql_query("""
        SELECT TICKER, date, RET as return
        FROM crsp_sp100_returns
        WHERE RET IS NOT NULL
        ORDER BY TICKER, date
    """, conn)
    
    df_crsp['date'] = pd.to_datetime(df_crsp['date'])
    df_crsp['return'] = pd.to_numeric(df_crsp['return'], errors='coerce')
    df_crsp = df_crsp[df_crsp['return'].notna()]
    
    conn.close()
    
    print(f"  Loaded {len(df_crsp):,} return records")
    print(f"  Date range: {df_crsp['date'].min()} to {df_crsp['date'].max()}")
    print(f"  Unique tickers: {df_crsp['TICKER'].nunique()}")
    
    return df_crsp


def get_crsp_cumulative_return(df_crsp, instrument, event_date, holding_days):
    """
    Get cumulative return from CRSP for an instrument from event_date.
    
    Args:
        df_crsp: DataFrame with CRSP returns (from load_crsp_returns)
        instrument: str - Instrument name (e.g., 'META.OQ')
        event_date: datetime - Event date
        holding_days: int - Number of holding days
        
    Returns:
        tuple: (cumulative_return, date_gap_days, crsp_ticker) or (None, None, None) if not available
    """
    # Check if in data gap
    if is_in_data_gap(instrument, event_date):
        return None, None, None
    
    # Get CRSP ticker
    ticker = get_crsp_ticker(instrument, event_date)
    if ticker is None:
        return None, None, None
    
    # Get returns from event date onwards
    returns_forward = df_crsp[
        (df_crsp['TICKER'] == ticker) & 
        (df_crsp['date'] >= event_date)
    ].sort_values('date').head(holding_days + 1)
    
    if len(returns_forward) < holding_days + 1:
        return None, None, ticker
    
    # Calculate date gap
    first_trading_date = returns_forward['date'].iloc[0]
    date_gap = (first_trading_date - event_date).days
    
    # Calculate cumulative return
    ret_cum = (1 + returns_forward['return'].iloc[1:holding_days+1]).prod() - 1
    
    return ret_cum, date_gap, ticker

