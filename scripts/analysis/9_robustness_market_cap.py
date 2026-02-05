"""
ROBUSTNESS TEST: Large Cap vs Small Cap Analysis
Analyzes whether ESG prediction accuracy and portfolio returns differ by firm size
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import shared styling (applies style on import)
from figure_style import COLORS, MODEL_COLORS, MODEL_LABELS

# Additional colors for this analysis
SIZE_COLORS = {
    'large': COLORS['primary'],
    'small': COLORS['accent'],
    'neutral': COLORS['neutral'],
}

print("\n" + "="*80)
print("ROBUSTNESS TEST: MARKET CAP ANALYSIS")
print("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

# Load main dataset
import utils
df = utils.load_working_data('period_dataset')
print(f"  Main dataset: {len(df):,} rows")

# Load market cap data (wide format: date as rows, instruments as columns)
print(f"  Loading market cap from: {config.MARKET_CAP_DB}")
conn_mcap = sqlite3.connect(config.MARKET_CAP_DB)
df_mcap_wide = pd.read_sql_query("SELECT * FROM market_cap_hard", conn_mcap)
conn_mcap.close()

# Convert wide to long format
df_mcap_wide['date'] = pd.to_datetime(df_mcap_wide['date']).dt.strftime('%Y-%m-%d')
instrument_cols = [c for c in df_mcap_wide.columns if c != 'date']
df_mcap = df_mcap_wide.melt(id_vars=['date'], value_vars=instrument_cols, 
                             var_name='instrument', value_name='market_cap')
df_mcap['market_cap'] = pd.to_numeric(df_mcap['market_cap'], errors='coerce')

df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

print(f"  Market cap data: {len(df_mcap):,} rows")
print(f"  Unique instruments in market cap: {df_mcap['instrument'].nunique()}")

# =============================================================================
# MERGE MARKET CAP DATA
# =============================================================================
print("\n[2] Merging market cap data...")

# Direct merge on instrument and date
df_merged = df.merge(
    df_mcap[['instrument', 'date', 'market_cap']], 
    left_on=['instrument', 'date_str'], 
    right_on=['instrument', 'date'],
    how='left',
    suffixes=('', '_mcap')
)

# Check merge success
n_matched = df_merged['market_cap'].notna().sum()
print(f"  Matched: {n_matched:,} / {len(df):,} ({100*n_matched/len(df):.1f}%)")

# =============================================================================
# CLASSIFY BY SIZE
# =============================================================================
print("\n[3] Classifying firms by size...")

# Filter to rating change days only (where we have miss calculations)
df_rc = df_merged[df_merged['days_to_next_release'] == 0].copy()
n_rc = len(df_rc)
print(f"  Rating change observations: {n_rc:,}")

# Calculate yearly quintiles based on market cap
df_rc['year'] = pd.to_datetime(df_rc['date']).dt.year

# For each year, classify into quintiles
def classify_size(group):
    if group['market_cap'].isna().all():
        group['size_quintile'] = np.nan
        group['size_class'] = np.nan
        return group
    
    group['size_quintile'] = pd.qcut(group['market_cap'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
    group['size_class'] = group['size_quintile'].map({
        1: 'Small',
        2: 'Small-Mid',
        3: 'Mid',
        4: 'Mid-Large',
        5: 'Large'
    })
    return group

df_rc = df_rc.groupby('year', group_keys=False).apply(classify_size)

# Summary stats
print("\n  Size Classification Summary:")
size_counts = df_rc['size_class'].value_counts()
for size, count in size_counts.items():
    print(f"    {size}: {count:,}")

n_with_size = df_rc['size_class'].notna().sum()
print(f"\n  Observations with size classification: {n_with_size:,} / {n_rc:,} ({100*n_with_size/n_rc:.1f}%)")

# =============================================================================
# ANALYZE PREDICTION ACCURACY BY SIZE
# =============================================================================
print("\n[4] Analyzing prediction accuracy by size...")

# Focus on Large (Q5) vs Small (Q1)
df_large = df_rc[df_rc['size_quintile'] == 5]
df_small = df_rc[df_rc['size_quintile'] == 1]

results = []

for model in ['base', 'base_a', 'ft', 'ft_a']:
    miss_col = config.get_miss_abs_col(model, 'simple')  # Use config function
    
    if miss_col not in df_rc.columns:
        print(f"  Warning: {miss_col} not found, skipping {model}")
        continue
    
    large_mae = df_large[miss_col].dropna().mean()
    small_mae = df_small[miss_col].dropna().mean()
    large_n = df_large[miss_col].notna().sum()
    small_n = df_small[miss_col].notna().sum()
    
    results.append({
        'Model': model,
        'Large Cap MAE': large_mae,
        'Large Cap N': large_n,
        'Small Cap MAE': small_mae,
        'Small Cap N': small_n,
        'Difference': small_mae - large_mae,
        'Pct Diff': 100 * (small_mae - large_mae) / large_mae if large_mae > 0 else np.nan
    })
    
    print(f"\n  {model.upper()}:")
    print(f"    Large Cap (Q5): MAE = {large_mae:.2f} (n={large_n:,})")
    print(f"    Small Cap (Q1): MAE = {small_mae:.2f} (n={small_n:,})")
    print(f"    Difference: {small_mae - large_mae:.2f} ({100*(small_mae-large_mae)/large_mae:.1f}%)")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(config.TABLES_DIR, 'robustness_market_cap_accuracy.csv'), index=False)

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[5] Creating visualization...")

if len(results) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = [r['Model'] for r in results]
    large_maes = [r['Large Cap MAE'] for r in results]
    small_maes = [r['Small Cap MAE'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, large_maes, width, label='Large Cap (Q5)', color=SIZE_COLORS['large'], edgecolor='white')
    bars2 = ax.bar(x + width/2, small_maes, width, label='Small Cap (Q1)', color=SIZE_COLORS['small'], edgecolor='white')
    
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_title('Prediction Accuracy by Firm Size', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=9, rotation=0, ha='center')
    ax.legend(frameon=True, fancybox=False, edgecolor='grey')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels with dynamic offset
    max_val = max(max(large_maes), max(small_maes))
    ax.set_ylim(0, max_val * 1.25)
    
    for bars, values in [(bars1, large_maes), (bars2, small_maes)]:
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig7_robustness_market_cap.png'), bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'fig7_robustness_market_cap.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Figure saved: fig7_robustness_market_cap.png/pdf")

# =============================================================================
# LATEX TABLE
# =============================================================================
print("\n[6] Creating LaTeX table...")

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Robustness: Prediction Accuracy by Firm Size}
\label{tab:robustness_size}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Large Cap (Q5)} & \multicolumn{2}{c}{Small Cap (Q1)} & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Model & MAE & N & MAE & N & Difference \\
\midrule
"""

# Use consistent MODEL_LABELS from figure_style (paper terminology: anchored)
model_labels_latex = {
    'base': 'Base',
    'base_a': 'Base + Anchored',
    'ft': 'Fine-tuned',
    'ft_a': 'FT + Anchored'
}

for r in results:
    label = model_labels_latex.get(r['Model'], r['Model'])
    latex_table += f"{label} & {r['Large Cap MAE']:.2f} & {int(r['Large Cap N'])} & {r['Small Cap MAE']:.2f} & {int(r['Small Cap N'])} & {r['Difference']:+.2f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Large Cap = Top quintile by market capitalization; Small Cap = Bottom quintile. MAE = Mean Absolute Error. Difference = Small Cap MAE - Large Cap MAE. Positive values indicate worse performance on small cap stocks.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(config.LATEX_DIR, 'table_robustness_market_cap.tex'), 'w') as f:
    f.write(latex_table)

print(f"  ✓ LaTeX table saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✓ MARKET CAP ROBUSTNESS TEST COMPLETE")
print("="*80)

print("\nKey Findings:")
if len(results) > 0:
    avg_diff = np.mean([r['Difference'] for r in results if not np.isnan(r['Difference'])])
    if avg_diff > 0:
        print(f"  • Models generally perform better on large cap stocks")
        print(f"  • Average MAE difference: {avg_diff:.2f} points higher for small caps")
    else:
        print(f"  • Models perform similarly across size categories")
        print(f"  • No significant size bias detected")
else:
    print("  • Insufficient data for market cap analysis")

