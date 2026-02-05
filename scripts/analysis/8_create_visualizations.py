"""
CREATE ALL VISUALIZATIONS FOR ESG PREDICTION PAPER
Publication-quality figures with professional styling (APA7 compatible)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import shared styling (applies style on import)
from figure_style import COLORS, MODEL_COLORS, MODEL_LABELS, MODEL_LABELS_SHORT, HEATMAP_CMAP, HEATMAP_VMIN, HEATMAP_VMAX

print("\n" + "="*80)
print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
print("="*80)

# Use main figures directory (PNG and PDF saved together)
FIG_DIR = config.FIGURES_DIR
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
import utils  # For load_working_data function
h1 = pd.read_csv(os.path.join(config.TABLES_DIR, 'h1_results.csv'), index_col=0)
h4 = pd.read_csv(os.path.join(config.TABLES_DIR, 'h4_portfolio_returns_comprehensive_crsp.csv'))
df_periods = utils.load_working_data('period_dataset')

# =============================================================================
# [FIGURE 1] Model Prediction Accuracy Comparison
# =============================================================================
print("\n[1] Creating Figure 1: Model Prediction Accuracy...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['base_simple', 'base_a_simple', 'ft_simple', 'ft_a_simple']
model_keys = ['base', 'base_a', 'ft', 'ft_a']
model_labels_short = [MODEL_LABELS_SHORT[k] for k in model_keys]
colors = [MODEL_COLORS[k] for k in model_keys]

mae_values = [h1.loc[m, 'MAE'] for m in models]

# Direction accuracy is now pre-computed in h1_results.csv by 2_h1_replication_fidelity.py
dir_values = []
for m in models:
    dir_val = h1.loc[m, 'Dir']
    # Handle NaN by falling back to 50% (random baseline)
    dir_values.append(dir_val if pd.notna(dir_val) else 50.0)

# Panel A: MAE
bars1 = axes[0].bar(model_labels_short, mae_values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('Mean Absolute Error (ESG Points)', fontweight='bold')
axes[0].set_title('Panel A: Prediction Accuracy', fontweight='bold', pad=15)
axes[0].set_ylim(0, max(mae_values) * 1.35)  # More padding for labels

# Add value labels on bars
for bar, val in zip(bars1, mae_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.03, 
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Panel B: Direction Accuracy
bars2 = axes[1].bar(model_labels_short, dir_values, color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_ylabel('Direction Accuracy (%)', fontweight='bold')
axes[1].set_title('Panel B: Direction Prediction', fontweight='bold', pad=15)
axes[1].axhline(y=50, color=COLORS['negative'], linestyle='--', linewidth=2, label='Random (50%)')
axes[1].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey')
axes[1].set_ylim(0, 75)

# Add value labels
for bar, val in zip(bars2, dir_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_model_comparison.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig1_model_comparison.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# [FIGURE 2] Portfolio Returns Heatmap (All Models)
# =============================================================================
print("\n[2] Creating Figure 2: Portfolio Returns Heatmap...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

signal_a = h4[h4['signal'] == 'signal_A']

for idx, model in enumerate(['base', 'base_a', 'ft', 'ft_a']):
    model_data = signal_a[signal_a['model'] == model]
    pivot = model_data.pivot(index='formation', columns='holding', values='ls_net') * 100
    
    # Reorder for logical presentation
    pivot = pivot.reindex(index=['t-1', 't-3', 't-5'], columns=['t+1', 't+3', 't+5'])
    
    # Create heatmap with colorblind-safe palette and consistent scale
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=HEATMAP_CMAP, center=0,
                cbar_kws={'label': 'Net Return (%)', 'shrink': 0.8},
                ax=axes[idx], linewidths=0.5, linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    
    axes[idx].set_title(f'{MODEL_LABELS[model]}', fontweight='bold', fontsize=12, pad=10)
    axes[idx].set_xlabel('Holding Period', fontweight='bold')
    axes[idx].set_ylabel('Formation Period', fontweight='bold')

plt.suptitle('Long-Short Portfolio Returns by Model\n(Net of 40 bps Transaction Costs)', 
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_portfolio_heatmaps.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig2_portfolio_heatmaps.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# [FIGURE 3] Best Configuration Comparison (t-3/t+5)
# =============================================================================
print("\n[3] Creating Figure 3: Best Configuration Returns...")

best_config = signal_a[(signal_a['formation'] == 't-3') & (signal_a['holding'] == 't+5')]

fig, ax = plt.subplots(figsize=(10, 5))

models_order = ['base', 'base_a', 'ft', 'ft_a']
x = np.arange(len(models_order))
width = 0.35

gross_returns = [best_config[best_config['model'] == m]['ls_gross'].values[0] * 100 for m in models_order]
net_returns = [best_config[best_config['model'] == m]['ls_net'].values[0] * 100 for m in models_order]
p_values = [best_config[best_config['model'] == m]['p_val'].values[0] for m in models_order]

bars_gross = ax.bar(x - width/2, gross_returns, width, label='Gross Return', 
                    color=COLORS['accent'], edgecolor='white', linewidth=1.5)
bars_net = ax.bar(x + width/2, net_returns, width, label='Net Return (after 40 bps)', 
                  color=COLORS['primary'], edgecolor='white', linewidth=1.5)

# Calculate y-limits with padding for labels
all_returns = gross_returns + net_returns
y_max = max(all_returns)
y_min = min(all_returns)
y_range = y_max - y_min if y_max != y_min else max(abs(y_max), 1)
ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.35)

# Add significance stars with proper offset
for i, (bar, p) in enumerate(zip(bars_net, p_values)):
    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    if star:
        offset = y_range * 0.08 if bar.get_height() >= 0 else -y_range * 0.08
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, 
                star, ha='center', va='bottom' if bar.get_height() >= 0 else 'top', 
                fontsize=12, fontweight='bold', color=COLORS['positive'])

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Return (%)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.set_title('Long-Short Portfolio Returns\n(Formation: t-3, Holding: t+5)', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in models_order], fontsize=9)
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='grey')

# Add note about significance - UPPER RIGHT corner
ax.text(0.98, 0.95, '* p<0.10  ** p<0.05  *** p<0.01', transform=ax.transAxes, 
        ha='right', va='top', fontsize=11, style='italic', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_best_config_returns.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig3_best_config_returns.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# [FIGURE 4] Long vs Short Portfolio Returns (Quintile Analysis)
# =============================================================================
print("\n[4] Creating Figure 4: Long vs Short Returns...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Focus on t-3/t+5 configuration (best performing)
config_data = signal_a[(signal_a['formation'] == 't-3') & (signal_a['holding'] == 't+5')]

for idx, model in enumerate(['base', 'base_a', 'ft', 'ft_a']):
    model_row = config_data[config_data['model'] == model].iloc[0]
    
    long_ret = model_row['long_ret'] * 100
    short_ret = model_row['short_ret'] * 100
    spread = long_ret - short_ret
    n_events = int(model_row['n_events'])
    
    bars = axes[idx].bar(['Long (Q5)\nHigh ESG Surprise', 'Short (Q1)\nLow ESG Surprise', 'L-S Spread'], 
                         [long_ret, short_ret, spread],
                         color=[COLORS['positive'], COLORS['negative'], COLORS['accent']],
                         edgecolor='white', linewidth=1.5)
    
    axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[idx].set_ylabel('Return (%)', fontweight='bold')
    axes[idx].set_title(f'{MODEL_LABELS[model]}\n(n={n_events:,} events)', fontweight='bold', fontsize=11)
    
    # Add value labels with dynamic offset based on data range
    y_range = max(abs(long_ret), abs(short_ret), abs(spread))
    label_offset = y_range * 0.08
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2, 
                       height + label_offset if height >= 0 else height - label_offset,
                       f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9, fontweight='bold')
    
    # Set y-limits with padding
    all_vals = [long_ret, short_ret, spread]
    axes[idx].set_ylim(min(all_vals) - y_range * 0.25, max(all_vals) + y_range * 0.25)

plt.suptitle('Portfolio Component Returns\n(Formation: t-3, Holding: t+5)', 
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_long_short_decomposition.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig4_long_short_decomposition.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# [FIGURE 5] Statistical Significance Overview
# =============================================================================
print("\n[5] Creating Figure 5: Statistical Significance...")

fig, ax = plt.subplots(figsize=(14, 18))  # Taller for better readability

# All configurations for signal_A
sig_data = signal_a[['model', 'formation', 'holding', 'ls_net', 'p_val', 't_stat']].copy()
sig_data['config'] = sig_data['formation'] + '/' + sig_data['holding']
sig_data['return_pct'] = sig_data['ls_net'] * 100
sig_data['significant'] = sig_data['p_val'] < 0.1

# Sort by return
sig_data = sig_data.sort_values('return_pct', ascending=False)

# Color by significance
colors_sig = [COLORS['positive'] if sig else COLORS['neutral'] for sig in sig_data['significant']]

bars = ax.barh(range(len(sig_data)), sig_data['return_pct'], color=colors_sig, edgecolor='white', linewidth=0.5)

# Add model labels with more space
for i, (idx, row) in enumerate(sig_data.iterrows()):
    model_label = MODEL_LABELS.get(row['model'], row['model'])
    label = f"{model_label} ({row['config']})"
    ax.text(-0.1, i, label, ha='right', va='center', fontsize=11)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Net Return (%)', fontweight='bold')
ax.set_title('Portfolio Returns Ranked by Performance\n(Green = Statistically Significant at 10%)', fontweight='bold')
ax.set_yticks([])

# Adjust x-axis to make room for labels
x_min = sig_data['return_pct'].min()
x_max = sig_data['return_pct'].max()
x_range = x_max - x_min
ax.set_xlim(x_min - x_range * 0.4, x_max + x_range * 0.15)  # More space on left for labels

# Add legend
significant_patch = mpatches.Patch(color=COLORS['positive'], label='Significant (p<0.10)')
neutral_patch = mpatches.Patch(color=COLORS['neutral'], label='Not Significant')
ax.legend(handles=[significant_patch, neutral_patch], loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_significance_overview.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig5_significance_overview.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# [FIGURE 6] In-Sample vs Out-of-Sample Performance
# =============================================================================
print("\n[6] Creating Figure 6: In-Sample vs Out-of-Sample...")

df_periods['year'] = pd.to_datetime(df_periods['date']).dt.year
# Data actually starts from 2017 (not 2014/2019)
min_year = df_periods['year'].min()
df_periods['sample'] = df_periods['year'].apply(lambda x: f'In-Sample\n({min_year}-2023)' if x < 2024 else 'Out-of-Sample\n(2024)')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Use correct column naming: base_simple_miss_absolute, ft_simple_miss_absolute
for idx, (model, model_name) in enumerate([('base', 'Gemini (Base)'), ('ft', 'Gemini (Fine-tuned)')]):
    # Try different column name patterns
    possible_cols = [
        f"{model}_simple_miss_absolute",
        f"{model}_miss_absolute", 
        f"{model}_c_simple_miss_absolute" if model == 'base' else f"{model}_simple_miss_absolute"
    ]
    mae_col = None
    for col in possible_cols:
        if col in df_periods.columns:
            mae_col = col
            break
    
    if mae_col and mae_col in df_periods.columns:
        # Filter to valid data only
        valid_data = df_periods[df_periods[mae_col].notna()]
        if len(valid_data) > 0:
            mae_by_sample = valid_data.groupby('sample')[mae_col].mean()
            
            bars = axes[idx].bar(mae_by_sample.index, mae_by_sample.values, 
                                color=[COLORS['primary'], COLORS['accent']], 
                                edgecolor='white', linewidth=1.5)
            
            axes[idx].set_ylabel('Mean Absolute Error', fontweight='bold')
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            
            # Add value labels with proper padding
            max_val = mae_by_sample.max()
            axes[idx].set_ylim(0, max_val * 1.25)
            for bar, val in zip(bars, mae_by_sample.values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
                              f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{model_name}', fontweight='bold')
    else:
        axes[idx].text(0.5, 0.5, f'Column not found\nTried: {possible_cols}', ha='center', va='center', 
                      transform=axes[idx].transAxes, fontsize=8)
        axes[idx].set_title(f'{model_name}', fontweight='bold')

plt.suptitle('Model Generalization: In-Sample vs Out-of-Sample Performance', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_sample_comparison.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'fig6_sample_comparison.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# GENERATE LATEX TABLES (APA7 Style)
# =============================================================================
print("\n[7] Creating LaTeX Tables (APA7 Style)...")

os.makedirs(config.LATEX_DIR, exist_ok=True)

# Table 1: Model Accuracy Summary
latex_table1 = r"""\begin{table}[htbp]
\centering
\caption{Model Prediction Accuracy Comparison}
\label{tab:model_accuracy}
\begin{tabular}{lcc}
\toprule
Model & MAE & Direction Accuracy (\%) \\
\midrule
"""

for m, label in [('base_simple', 'Gemini (Base)'), ('base_a_simple', 'Gemini (Base) + Anchored'), 
                 ('ft_simple', 'Gemini (Fine-tuned)'), ('ft_a_simple', 'Gemini (FT) + Anchored')]:
    mae = h1.loc[m, 'MAE']
    dir_acc = h1.loc[m, 'Dir']
    latex_table1 += f"{label} & {mae:.2f} & {dir_acc:.1f} \\\\\n"

latex_table1 += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} MAE = Mean Absolute Error in ESG score points. Direction Accuracy measures the percentage of correctly predicted ESG score direction changes. Sample includes """ + f"{len(df_periods[df_periods['fair_comparison_sample']==1]):,}" + r""" rating change observations where all models have valid predictions.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(config.LATEX_DIR, 'table1_model_accuracy.tex'), 'w') as f:
    f.write(latex_table1)

# Table 2: Portfolio Returns
latex_table2 = r"""\begin{table}[htbp]
\centering
\caption{Long-Short Portfolio Returns by Model and Horizon}
\label{tab:portfolio_returns}
\begin{tabular}{llcccccc}
\toprule
Model & Formation & \multicolumn{2}{c}{t+1} & \multicolumn{2}{c}{t+3} & \multicolumn{2}{c}{t+5} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
& & Gross & Net & Gross & Net & Gross & Net \\
\midrule
"""

for model in ['base', 'base_a', 'ft', 'ft_a']:
    model_data = signal_a[signal_a['model'] == model]
    for i, form in enumerate(['t-1', 't-3', 't-5']):
        row_data = model_data[model_data['formation'] == form]
        if i == 0:
            latex_table2 += f"{MODEL_LABELS[model]} & {form} "
        else:
            latex_table2 += f" & {form} "
        
        for hold in ['t+1', 't+3', 't+5']:
            cell = row_data[row_data['holding'] == hold]
            if len(cell) > 0:
                gross = cell['ls_gross'].values[0] * 100
                net = cell['ls_net'].values[0] * 100
                p = cell['p_val'].values[0]
                star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                latex_table2 += f"& {gross:.2f} & {net:.2f}{star} "
            else:
                latex_table2 += "& -- & -- "
        latex_table2 += "\\\\\n"
    latex_table2 += "\\midrule\n"

latex_table2 = latex_table2.rstrip("\\midrule\n") + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Returns are expressed in percentage points. Gross returns are before transaction costs; Net returns are after 40 basis points round-trip costs. * p < .10, ** p < .05, *** p < .01 (two-tailed t-test).
\end{tablenotes}
\end{table}
"""

with open(os.path.join(config.LATEX_DIR, 'table2_portfolio_returns.tex'), 'w') as f:
    f.write(latex_table2)

# Table 3: Summary Statistics
latex_table3 = r"""\begin{table}[htbp]
\centering
\caption{Summary Statistics}
\label{tab:summary_stats}
\begin{tabular}{lcccc}
\toprule
Variable & Mean & SD & Min & Max \\
\midrule
"""

# Add summary stats for key variables
for col, label in [('base_miss_absolute', 'Base Model MAE'),
                   ('ft_miss_absolute', 'Fine-tuned Model MAE'),
                   ('esg_score', 'Actual ESG Score')]:
    if col in df_periods.columns:
        data = df_periods[col].dropna()
        latex_table3 += f"{label} & {data.mean():.2f} & {data.std():.2f} & {data.min():.2f} & {data.max():.2f} \\\\\n"

latex_table3 += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} MAE = Mean Absolute Error. Statistics computed on fair comparison sample where all models have valid predictions.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(config.LATEX_DIR, 'table3_summary_stats.tex'), 'w') as f:
    f.write(latex_table3)

print(f"  ✓ LaTeX tables saved to: {config.LATEX_DIR}")

# =============================================================================
# [APPENDIX FIGURES] Additional figures for appendix
# =============================================================================
print("\n[8] Creating Appendix Figures...")

# --- h1_insample_vs_outsample_base.png ---
print("  Creating h1_insample_vs_outsample_base.png...")
fig, ax = plt.subplots(figsize=(8, 4))
df_periods['year'] = pd.to_datetime(df_periods['date']).dt.year
# Data actually starts from 2017
min_year = df_periods['year'].min()
df_periods['sample'] = df_periods['year'].apply(lambda x: f'In-Sample\n({min_year}-2023)' if x < 2024 else 'Out-of-Sample\n(2024)')

# Try different column name patterns for base model
mae_col = None
for col in ["base_simple_miss_absolute", "base_miss_absolute", "base_a_simple_miss_absolute"]:
    if col in df_periods.columns:
        mae_col = col
        break

if mae_col and df_periods[mae_col].notna().sum() > 0:
    valid_data = df_periods[df_periods[mae_col].notna()]
    mae_by_sample = valid_data.groupby('sample')[mae_col].mean()
    bars = ax.bar(mae_by_sample.index, mae_by_sample.values, 
                  color=[COLORS['primary'], COLORS['accent']], 
                  edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_title('Base Model: In-Sample vs Out-of-Sample Performance', fontweight='bold')
    max_val = mae_by_sample.max()
    ax.set_ylim(0, max_val * 1.25)
    for bar, val in zip(bars, mae_by_sample.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No MAE data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Base Model: In-Sample vs Out-of-Sample Performance', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'h1_insample_vs_outsample_base.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'h1_insample_vs_outsample_base.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# --- h1_insample_vs_outsample_ft.png ---
print("  Creating h1_insample_vs_outsample_ft.png...")
fig, ax = plt.subplots(figsize=(8, 4))

# Try different column name patterns for ft model
mae_col = None
for col in ["ft_simple_miss_absolute", "ft_miss_absolute", "ft_a_simple_miss_absolute"]:
    if col in df_periods.columns:
        mae_col = col
        break

if mae_col and df_periods[mae_col].notna().sum() > 0:
    valid_data = df_periods[df_periods[mae_col].notna()]
    mae_by_sample = valid_data.groupby('sample')[mae_col].mean()
    bars = ax.bar(mae_by_sample.index, mae_by_sample.values, 
                  color=[COLORS['primary'], COLORS['accent']], 
                  edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax.set_title('Fine-Tuned Model: In-Sample vs Out-of-Sample Performance', fontweight='bold')
    max_val = mae_by_sample.max()
    ax.set_ylim(0, max_val * 1.25)
    for bar, val in zip(bars, mae_by_sample.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No MAE data available', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Fine-Tuned Model: In-Sample vs Out-of-Sample Performance', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'h1_insample_vs_outsample_ft.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'h1_insample_vs_outsample_ft.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# --- h4_insample_vs_outsample_returns.png ---
print("  Creating h4_insample_vs_outsample_returns.png...")

# Load in-vs-out returns data if available
h4_in_out_path = os.path.join(config.TABLES_DIR, 'h4_in_vs_out_sample_returns.csv')
if os.path.exists(h4_in_out_path):
    df_in_out = pd.read_csv(h4_in_out_path)
    
    # Create side-by-side bar chart (like MAE comparison)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, model in enumerate(['base', 'ft']):
        ax = axes[idx]
        model_data = df_in_out[df_in_out['model'] == model]
        
        if len(model_data) >= 2:
            # Get in-sample and out-of-sample values
            in_sample_data = model_data[model_data['sample'] == 'in_sample']
            out_sample_data = model_data[model_data['sample'] == 'out_of_sample']
            
            if len(in_sample_data) > 0 and len(out_sample_data) > 0:
                in_ret = in_sample_data['ls_net'].values[0] * 100
                out_ret = out_sample_data['ls_net'].values[0] * 100
                in_p = in_sample_data['p_val'].values[0]
                out_p = out_sample_data['p_val'].values[0]
                in_n = int(in_sample_data['n_events'].values[0])
                out_n = int(out_sample_data['n_events'].values[0])
                
                # Determine min year from df_periods for label
                min_year = df_periods['year'].min()
                
                x_labels = [f'In-Sample\n({min_year}-2023)\nN={in_n}', f'Out-of-Sample\n(2024)\nN={out_n}']
                returns = [in_ret, out_ret]
                p_vals = [in_p, out_p]
                
                bars = ax.bar(x_labels, returns, 
                              color=[COLORS['primary'], COLORS['accent']], 
                              edgecolor='white', linewidth=1.5)
                
                # Add value labels with significance stars
                max_val = max(abs(min(returns)), abs(max(returns)))
                y_range = max_val if max_val > 0 else 1
                ax.set_ylim(min(returns) - y_range * 0.3, max(returns) + y_range * 0.4)
                
                for bar, ret, p in zip(bars, returns, p_vals):
                    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                    label_y = bar.get_height() + y_range * 0.08 if bar.get_height() >= 0 else bar.get_height() - y_range * 0.15
                    va = 'bottom' if bar.get_height() >= 0 else 'top'
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, 
                            f'{ret:+.2f}%{star}', ha='center', va=va, fontsize=11, fontweight='bold')
                
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax.set_ylabel('Net Return (%)', fontweight='bold')
                
                model_name = 'Base Model' if model == 'base' else 'Fine-Tuned Model'
                ax.set_title(f'{model_name}', fontweight='bold')
                
                # Add significance legend
                ax.text(0.98, 0.02, '* p<0.10  ** p<0.05  *** p<0.01', 
                        transform=ax.transAxes, ha='right', va='bottom', 
                        fontsize=9, style='italic', color='gray')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Economic Value: In-Sample vs Out-of-Sample Performance\n(Formation: t-3, Holding: t+5)', 
                 fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'h4_insample_vs_outsample_returns.png'), bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(FIG_DIR, 'h4_insample_vs_outsample_returns.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print("    ✓ Created h4_insample_vs_outsample_returns figure")
else:
    print("    ⚠ h4_in_vs_out_sample_returns.csv not found - run 5e_h4_in_vs_out_robustness.py first")

# --- h4_returns_by_model.png ---
print("  Creating h4_returns_by_model.png...")
fig, ax = plt.subplots(figsize=(10, 6))

# Best configuration (t-3/t+5) returns by model
best_config = signal_a[(signal_a['formation'] == 't-3') & (signal_a['holding'] == 't+5')]
models_order = ['base', 'base_a', 'ft', 'ft_a']
net_returns = [best_config[best_config['model'] == m]['ls_net'].values[0] * 100 for m in models_order]
p_values = [best_config[best_config['model'] == m]['p_val'].values[0] for m in models_order]
colors_list = [MODEL_COLORS[m] for m in models_order]
labels = [MODEL_LABELS[m] for m in models_order]

bars = ax.bar(labels, net_returns, color=colors_list, edgecolor='white', linewidth=1.5)

# Add significance markers with proper spacing
max_val = max(net_returns)
min_val = min(net_returns)
y_range = max_val - min_val if max_val != min_val else max(abs(max_val), 1)
ax.set_ylim(min_val - y_range * 0.15, max_val + y_range * 0.35)

for bar, p, ret in zip(bars, p_values, net_returns):
    star = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    label_y = bar.get_height() + y_range * 0.05 if bar.get_height() >= 0 else bar.get_height() - y_range * 0.1
    va = 'bottom' if bar.get_height() >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, label_y, 
            f'{ret:.2f}%{star}', ha='center', va=va, fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('Net Return (%)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
ax.set_title('Portfolio Returns by Model\n(Formation: t-3, Holding: t+5, Net of 40 bps)', fontweight='bold')
ax.text(0.98, 0.95, '* p<0.10  ** p<0.05  *** p<0.01', transform=ax.transAxes, 
        ha='right', va='top', fontsize=11, style='italic', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'h4_returns_by_model.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'h4_returns_by_model.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# --- h1_model_comparison.png (legacy compatibility) ---
print("  Creating h1_model_comparison.png...")
fig, ax = plt.subplots(figsize=(10, 6))
mae_values = [h1.loc[m, 'MAE'] for m in models]
# Narrower bars (width=0.5) so labels fit inside the plot area
x_pos = np.arange(len(model_labels_short))
bars = ax.bar(x_pos, mae_values, width=0.5, color=colors, edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_labels_short)
ax.set_ylabel('Mean Absolute Error (ESG Points)', fontweight='bold')
ax.set_title('Model Prediction Accuracy Comparison', fontweight='bold')
# Add more y-axis padding for labels
max_val = max(mae_values)
ax.set_ylim(0, max_val * 1.2)
for bar, val in zip(bars, mae_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02, 
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'h1_model_comparison.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'h1_model_comparison.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# --- h4_portfolio_returns_heatmap.png (legacy compatibility) ---
print("  Creating h4_portfolio_returns_heatmap.png...")
fig, ax = plt.subplots(figsize=(10, 6))
# Just FT model heatmap
ft_data = signal_a[signal_a['model'] == 'ft']
pivot = ft_data.pivot(index='formation', columns='holding', values='ls_net') * 100
pivot = pivot.reindex(index=['t-1', 't-3', 't-5'], columns=['t+1', 't+3', 't+5'])
sns.heatmap(pivot, annot=True, fmt='.2f', cmap=HEATMAP_CMAP, center=0,
            cbar_kws={'label': 'Net Return (%)', 'shrink': 0.8},
            ax=ax, linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 12, 'fontweight': 'bold'},
            vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
ax.set_title('Fine-Tuned Model: Portfolio Returns Heatmap\n(Net of 40 bps)', fontweight='bold')
ax.set_xlabel('Holding Period', fontweight='bold')
ax.set_ylabel('Formation Period', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'h4_portfolio_returns_heatmap.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(FIG_DIR, 'h4_portfolio_returns_heatmap.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ Appendix figures created")

# =============================================================================
# SUMMARY OUTPUT
# =============================================================================
print("\n" + "="*80)
print("✓ VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nFigures saved to: {FIG_DIR}")
print(f"LaTeX tables saved to: {config.LATEX_DIR}")
print("\nGenerated files:")
print("  - fig1_model_comparison     (Model accuracy comparison)")
print("  - fig2_portfolio_heatmaps   (Returns heatmap by model)")
print("  - fig3_best_config_returns  (Best configuration: t-3/t+5)")
print("  - fig4_long_short_decomposition (Long vs Short breakdown)")
print("  - fig5_significance_overview (Significance ranking)")
print("  - fig6_sample_comparison    (In-sample vs out-of-sample)")
print("\nLaTeX tables:")
print("  - table1_model_accuracy.tex")
print("  - table2_portfolio_returns.tex")
print("  - table3_summary_stats.tex")
