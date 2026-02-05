"""
SHARED FIGURE STYLING FOR ESG PREDICTION PAPER
Import this module in all visualization scripts to ensure consistency.
All figures will use:
- Times New Roman font (matches LaTeX paper)
- Colorblind-safe Okabe-Ito palette
- Professional finance journal styling
"""
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# COLORBLIND-SAFE COLOR PALETTE (Okabe-Ito + Grey)
# =============================================================================
COLORS = {
    'primary': '#2C3E50',      # Dark blue-grey (text, main elements)
    'secondary': '#34495E',    # Medium blue-grey
    'accent': '#0072B2',       # Colorblind-safe blue (Okabe-Ito)
    'accent2': '#56B4E9',      # Colorblind-safe light blue
    'positive': '#009E73',     # Colorblind-safe green (Okabe-Ito)
    'negative': '#D55E00',     # Colorblind-safe orange-red (Okabe-Ito)
    'neutral': '#95A5A6',      # Grey
    'light': '#ECF0F1',        # Light grey
    'text': '#2C3E50',         # Dark text
    'warning': '#F0E442',      # Colorblind-safe yellow
}

# Model-specific colors (colorblind-safe)
MODEL_COLORS = {
    'base': '#999999',         # Grey
    'base_a': '#666666',       # Darker grey (anchored)
    'ft': '#0072B2',           # Colorblind-safe blue
    'ft_a': '#005587',         # Darker blue (anchored)
}

# Model labels for figures
MODEL_LABELS = {
    'base': 'Base',
    'base_a': 'Base + Anchored',
    'ft': 'Fine-tuned',
    'ft_a': 'FT + Anchored',
}

# Short labels for bar charts
MODEL_LABELS_SHORT = {
    'base': 'Base',
    'base_a': 'Base +\nAnchored',
    'ft': 'Fine-\ntuned',
    'ft_a': 'FT +\nAnchored',
}

# Colorblind-safe diverging colormap for heatmaps
# 'PuOr' (Purple-Orange) is colorblind-safe for diverging data
HEATMAP_CMAP = 'PuOr'
HEATMAP_VMIN = -2.5
HEATMAP_VMAX = 1.5


def apply_style():
    """
    Apply consistent matplotlib style across all figures.
    Call this at the start of any visualization script.
    """
    # Matplotlib configuration - matches LaTeX paper font
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': '#2C3E50',
        'axes.labelcolor': '#2C3E50',
        'xtick.color': '#2C3E50',
        'ytick.color': '#2C3E50',
        'text.color': '#2C3E50',
    })
    
    # Seaborn styling
    sns.set_style("white")

