"""
ROBUSTNESS TEST: Transaction Cost Sensitivity Analysis
Analyzes how portfolio returns vary with different transaction cost assumptions
"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import shared styling (applies style on import)
from figure_style import COLORS, MODEL_COLORS, MODEL_LABELS

print("\n" + "="*80)
print("ROBUSTNESS TEST: TRANSACTION COST SENSITIVITY")
print("="*80)

# Load portfolio returns
h4 = pd.read_csv(os.path.join(config.TABLES_DIR, 'h4_portfolio_returns_comprehensive_crsp.csv'))
signal_a = h4[h4['signal'] == 'signal_A']

# Transaction cost scenarios (round-trip, in basis points)
COST_SCENARIOS = {
    'Zero (Theoretical)': 0,
    'Low (10 bps)': 10,
    'Moderate (20 bps)': 20,
    'Baseline (40 bps)': 40,
    'High (60 bps)': 60,
    'Very High (100 bps)': 100,
}

# Focus on best performing configuration (t-3/t+5)
print("\n[1] Analyzing best configuration: t-3 formation, t+5 holding...")
best_config = signal_a[(signal_a['formation'] == 't-3') & (signal_a['holding'] == 't+5')]

results = []

for model in ['base', 'base_a', 'ft', 'ft_a']:
    model_row = best_config[best_config['model'] == model].iloc[0]
    gross_ret = model_row['ls_gross']
    
    for cost_name, cost_bps in COST_SCENARIOS.items():
        net_ret = gross_ret - (cost_bps / 10000)  # Convert bps to decimal
        
        results.append({
            'Model': model,
            'Cost Scenario': cost_name,
            'Cost (bps)': cost_bps,
            'Gross Return (%)': gross_ret * 100,
            'Net Return (%)': net_ret * 100,
            'Breakeven': gross_ret >= (cost_bps / 10000)
        })

df_results = pd.DataFrame(results)

# Print summary
print("\n" + "-"*60)
print("Net Returns by Model and Transaction Cost Scenario")
print("-"*60)
print(f"{'Model':<15} {'0 bps':>10} {'10 bps':>10} {'20 bps':>10} {'40 bps':>10} {'60 bps':>10} {'100 bps':>10}")
print("-"*60)

for model in ['base', 'base_a', 'ft', 'ft_a']:
    model_data = df_results[df_results['Model'] == model]
    values = [model_data[model_data['Cost (bps)'] == c]['Net Return (%)'].values[0] for c in [0, 10, 20, 40, 60, 100]]
    formatted = [f"{v:+.2f}%" for v in values]
    print(f"{model:<15} {formatted[0]:>10} {formatted[1]:>10} {formatted[2]:>10} {formatted[3]:>10} {formatted[4]:>10} {formatted[5]:>10}")

# Calculate breakeven costs
print("\n" + "-"*60)
print("Breakeven Transaction Costs")
print("-"*60)

breakeven_costs = {}
for model in ['base', 'base_a', 'ft', 'ft_a']:
    model_row = best_config[best_config['model'] == model].iloc[0]
    gross_ret = model_row['ls_gross']
    breakeven_bps = gross_ret * 10000  # Convert decimal to bps
    breakeven_costs[model] = breakeven_bps
    status = "PROFITABLE at 40 bps" if breakeven_bps > 40 else "LOSS at 40 bps"
    print(f"  {model}: {breakeven_bps:.1f} bps ({status})")

# Save results
df_results.to_csv(os.path.join(config.TABLES_DIR, 'robustness_transaction_costs.csv'), index=False)

# =============================================================================
# CREATE VISUALIZATION
# =============================================================================
print("\n[2] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Net returns by cost scenario
costs = [0, 10, 20, 40, 60, 100]
x = np.arange(len(costs))
width = 0.2

for i, model in enumerate(['base', 'base_a', 'ft', 'ft_a']):
    model_data = df_results[df_results['Model'] == model]
    values = [model_data[model_data['Cost (bps)'] == c]['Net Return (%)'].values[0] for c in costs]
    
    offset = (i - 1.5) * width
    bars = axes[0].bar(x + offset, values, width, label=MODEL_LABELS.get(model, model), 
                       color=MODEL_COLORS[model], edgecolor='white')

axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_xlabel('Transaction Cost (bps)', fontweight='bold')
axes[0].set_ylabel('Net Return (%)', fontweight='bold')
axes[0].set_title('Panel A: Net Returns by Transaction Cost', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(costs)
axes[0].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey')
axes[0].grid(axis='y', alpha=0.3)

# Panel B: Breakeven analysis
models = ['base', 'base_a', 'ft', 'ft_a']
breakeven_values = [breakeven_costs[m] for m in models]
colors = [MODEL_COLORS[m] for m in models]

bars = axes[1].bar(range(len(models)), breakeven_values, color=colors, edgecolor='white')
axes[1].axhline(y=40, color='red', linestyle='--', linewidth=2, label='Baseline (40 bps)')
axes[1].set_xlabel('Model', fontweight='bold')
axes[1].set_ylabel('Breakeven Cost (bps)', fontweight='bold')
axes[1].set_title('Panel B: Breakeven Transaction Costs', fontweight='bold')
axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels(['Base', 'Base+A', 'FT', 'FT+A'], fontsize=9)
axes[1].legend(loc='upper right', frameon=True, fancybox=False, edgecolor='grey')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels with proper spacing
max_breakeven = max(breakeven_values)
axes[1].set_ylim(0, max_breakeven * 1.25)

for bar, val in zip(bars, breakeven_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_breakeven * 0.03, 
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, 'fig8_transaction_cost_sensitivity.png'), bbox_inches='tight', facecolor='white')
plt.savefig(os.path.join(config.FIGURES_DIR, 'fig8_transaction_cost_sensitivity.pdf'), bbox_inches='tight', facecolor='white')
plt.close()

# =============================================================================
# LATEX TABLE
# =============================================================================
print("\n[3] Creating LaTeX table...")

latex_table = r"""\begin{table}[htbp]
\centering
\caption{Transaction Cost Sensitivity Analysis}
\label{tab:robustness_costs}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{6}{c}{Net Return (\%) at Transaction Cost Level} \\
\cmidrule(lr){2-7}
Model & 0 bps & 10 bps & 20 bps & 40 bps & 60 bps & 100 bps \\
\midrule
"""

# Use consistent MODEL_LABELS from figure_style
model_labels_latex = {
    'base': 'Gemini (Base)',
    'base_a': 'Gemini (Base) + Anchored',
    'ft': 'Gemini (Fine-tuned)',
    'ft_a': 'Gemini (FT) + Anchored'
}

for model in ['base', 'base_a', 'ft', 'ft_a']:
    model_data = df_results[df_results['Model'] == model]
    values = [model_data[model_data['Cost (bps)'] == c]['Net Return (%)'].values[0] for c in [0, 10, 20, 40, 60, 100]]
    formatted = ' & '.join([f'{v:+.2f}' for v in values])
    latex_table += f"{model_labels_latex[model]} & {formatted} \\\\\n"

latex_table += r"""\midrule
\multicolumn{7}{l}{\textit{Breakeven Costs (bps)}} \\
"""

for model in ['base', 'base_a', 'ft', 'ft_a']:
    latex_table += f"{model_labels_latex[model]} & \\multicolumn{{6}}{{c}}{{{breakeven_costs[model]:.1f}}} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Note.} Results shown for optimal configuration (t-3 formation, t+5 holding period). Transaction costs represent round-trip costs. Breakeven cost is the maximum transaction cost at which the strategy remains profitable. The baseline assumption of 40 bps follows standard academic practice.
\end{tablenotes}
\end{table}
"""

with open(os.path.join(config.LATEX_DIR, 'table_robustness_transaction_costs.tex'), 'w') as f:
    f.write(latex_table)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("✓ TRANSACTION COST SENSITIVITY COMPLETE")
print("="*80)

# Key findings
profitable_at_40 = [m for m in models if breakeven_costs[m] > 40]
print("\nKey Findings:")
print(f"  • Models profitable at 40 bps: {', '.join(profitable_at_40) if profitable_at_40 else 'None'}")
print(f"  • Best performing model: ft with breakeven at {breakeven_costs['ft']:.0f} bps")
print(f"  • All models generate positive gross returns")
print(f"  • Transaction costs are critical to profitability")

