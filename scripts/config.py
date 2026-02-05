"""
Configuration file for ESG Prediction Analysis.

Contains all paths, parameters, and settings for the analysis pipeline.
Configured for portability - users must update proprietary data paths.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Base directory (automatically resolved relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # Parent of scripts/

# Data directory (contains included data files)
DATA_DIR_ROOT = os.path.join(BASE_DIR, 'data')

# Results directory (outputs go here)
RESULTS_DIR_ROOT = os.path.join(BASE_DIR, 'results')

# Main database - gemini_esg_data.db (unified database with all sample flags)
# Contains: full_sample, fair_comparison_sample, h3_sample, is_2024_test flags
# This is the primary data source for all analyses
MAIN_DB = os.path.join(DATA_DIR_ROOT, 'gemini_esg_data.db')

# Fama-French factors file (included in repository)
FF_FACTORS_FILE = os.path.join(DATA_DIR_ROOT, 'F-F_Research_Data_5_Factors_2x3_daily.xlsx')

# =============================================================================
# PROPRIETARY DATA PATHS (User must configure)
# =============================================================================
# These files contain proprietary data (CRSP, Refinitiv) and are NOT included
# in the GitHub repository due to licensing restrictions.

PROPRIETARY_DATA_DIR = os.path.join(DATA_DIR_ROOT, 'proprietary___not_available')

SECTORS_DB = os.path.join(PROPRIETARY_DATA_DIR, 'S&P100_sectors_hard.sqlite')
MARKET_CAP_DB = os.path.join(PROPRIETARY_DATA_DIR, 'S&P100_market_cap_hard.sqlite')
CRSP_DB_PATH = os.path.join(PROPRIETARY_DATA_DIR, 'CRSP_SP100_ReturnData.db')

# =============================================================================
# SAMPLE DEFINITIONS (from gemini_esg_data.db)
# =============================================================================
# 
# SAMPLE FLAGS IN DATABASE:
# - full_sample = 1: Rating changes with base_miss, excludes PLTR/PYPL (N=938)
# - fair_comparison_sample = 1: All 4 models have predictions (N=748)  
# - h3_sample = 1: Has cert + length + controversies data (N=933)
# - is_2024_test = 1: 2024 calendar year (out-of-sample)
# - is_rating_change = 1: Rating change event day
# - is_excluded_instrument = 1: BLK, PLTR, PYPL
#
# WHEN TO USE WHICH:
# - PRIMARY_SAMPLE: Use for ALL main analyses (H1, H2, H3, H4) - ensures consistent N
# - full_sample: Only for robustness checks comparing base/base_a/ft without ft_a
#
# PRIMARY_SAMPLE = fair_comparison_sample AND h3_sample
# This ensures: all 4 models have predictions AND cert/length data available
#
USE_PRIMARY_SAMPLE = True  # Main analyses use primary_sample (N=747)

# Sample counts (from gemini_esg_data.db)
# 
# KEY INSIGHT: Some rating change days have multiple document rows.
# Use is_primary_event = 1 for unique events (720), not primary_sample rows (747)
#
SAMPLE_COUNTS = {
    # PRIMARY ANALYSIS SAMPLE (use for ALL hypotheses)
    'is_primary_event': 720,         # Unique rating change events
    'is_primary_event_2024': 118,    # Out-of-sample test period
    
    # INTERMEDIATE SAMPLES (for reference)
    'primary_sample_rows': 747,      # Rows meeting all criteria (may have duplicates)
    'fair_comparison_sample': 748,   # All 4 models have predictions
    'h3_sample': 933,                # Has cert/length/controversies
    'full_sample': 938,              # base/base_a/ft only (robustness)
}

# Output directory (within repository structure)
OUTPUT_DIR = os.path.join(RESULTS_DIR_ROOT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subdirectories for outputs
TABLES_DIR = os.path.join(OUTPUT_DIR, 'csv')       # CSV output tables
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')  # PNG and PDF figures
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')        # Intermediate data files
LATEX_DIR = os.path.join(OUTPUT_DIR, 'latex')      # LaTeX table outputs

# Working data database (intermediate datasets for analysis)
WORKING_DATA_DB = os.path.join(DATA_DIR, 'working_data.db')

# Flag for fair comparison sample (used by utils.py)
USE_FAIR_COMPARISON_SAMPLE = True

for dir_path in [TABLES_DIR, FIGURES_DIR, DATA_DIR, LATEX_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# EXCLUSIONS
# =============================================================================

# Instruments to exclude (per user specifications)
# BLK.N: No estimations available
# PLTR.OQ and PYPL.OQ: Non-clean data, not in S&P500 dataset
EXCLUDED_INSTRUMENTS = ['BLK.N', 'PLTR.OQ', 'PYPL.OQ']

# =============================================================================
# SAMPLE SPLITS
# =============================================================================

# In-sample (training): 2014-2023
# Out-of-sample (test/post knowledge cutoff): 2024
IN_SAMPLE_START = '2014-01-01'
IN_SAMPLE_END = '2023-12-31'
OUT_OF_SAMPLE_START = '2024-01-01'
OUT_OF_SAMPLE_END = '2024-12-31'

# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

# Model types (a = anchored with prior-period score)
BASE_MODELS = ['base', 'base_a']
FT_MODELS = ['ft', 'ft_a']
ALL_MODELS = BASE_MODELS + FT_MODELS

# Rolling average types
ROLLING_AVG_TYPES = ['simple', 'cert', 'length']

# All model-rolling avg combinations (12 total)
ALL_MODEL_COMBINATIONS = [
    f"{model}_{roltype}" 
    for model in ALL_MODELS 
    for roltype in ROLLING_AVG_TYPES
]

# Model names for display
MODEL_DISPLAY_NAMES = {
    'base': 'Base',
    'base_a': 'Base + Anchored',
    'ft': 'Fine-Tuned',
    'ft_a': 'Fine-Tuned + Anchored'
}

ROLLING_AVG_DISPLAY_NAMES = {
    'simple': 'Simple Average',
    'cert': 'Certainty-Weighted',
    'length': 'Length-Weighted'
}

# =============================================================================
# SUPERSECTOR MAPPING (ICB hierarchical structure)
# =============================================================================

SUPERSECTOR_MAPPING = {
    # Financials
    'Banking Services': 'Financials',
    'Investment Banking & Investment Services': 'Financials',
    'Insurance': 'Financials',
    'Residential & Commercial REITs': 'Financials',

    # Technology
    'Software & IT Services': 'Technology',
    'Semiconductors & Semiconductor Equipment': 'Technology',
    'Computers, Phones & Household Electronics': 'Technology',
    'Communications & Networking': 'Technology',

    # Consumer & Retail
    'Beverages': 'Consumer & Retail',
    'Food & Tobacco': 'Consumer & Retail',
    'Textiles & Apparel': 'Consumer & Retail',
    'Specialty Retailers': 'Consumer & Retail',
    'Diversified Retail': 'Consumer & Retail',
    'Food & Drug Retailing': 'Consumer & Retail',
    'Hotels & Entertainment Services': 'Consumer & Retail',
    'Personal & Household Products & Services': 'Consumer & Retail',
    'Consumer Goods Conglomerates': 'Consumer & Retail',
    'Media & Publishing': 'Consumer & Retail',

    # Healthcare
    'Healthcare Equipment & Supplies': 'Healthcare',
    'Healthcare Providers & Services': 'Healthcare',
    'Pharmaceuticals': 'Healthcare',

    # Industrials
    'Aerospace & Defense': 'Industrials',
    'Machinery, Tools, Heavy Vehicles, Trains & Ships': 'Industrials',
    'Freight & Logistics Services': 'Industrials',
    'Professional & Commercial Services': 'Industrials',
    'Chemicals': 'Industrials',

    # Energy & Utilities
    'Oil & Gas': 'Energy & Utilities',
    'Electric Utilities & IPPs': 'Energy & Utilities',
    'Automobiles & Auto Parts': 'Energy & Utilities',
    
    # Telecommunications (added for complete coverage)
    'Telecommunications Services': 'Technology',
    
    # Financial Technology (added for complete coverage)
    'Financial Technology (Fintech) & Infrastructure': 'Financials'
}

# =============================================================================
# EVENT STUDY PARAMETERS
# =============================================================================

# Formation periods (days before ESG release at t=0)
FORMATION_PERIODS = [1, 3, 5]  # t-1, t-3, t-5

# Holding periods (days after ESG release at t=0)
HOLDING_PERIODS = [1, 3, 5]  # t+1, t+3, t+5

# Transaction costs (basis points per trade)
# Total cost = 4 * 10bps = 40bps (open long, close long, open short, close short)
TRANSACTION_COST_BPS = 10  # 10 bps per way

# Number of quintiles for portfolio sorting
N_QUINTILES = 5

# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

# Significance levels
ALPHA_LEVELS = [0.01, 0.05]  # 1% and 5%

# Standard error clustering
# Currently using firm-level clustering
# TODO: Consider adding two-way clustering (firm + date) or event-level clustering
SE_CLUSTER_METHOD = 'firm'

# Multiple testing correction methods
MULTIPLE_TEST_METHODS = ['bonferroni', 'benjamini-hochberg']

# =============================================================================
# PLOTTING PARAMETERS
# =============================================================================

# Figure settings
FIG_DPI = 300
FIG_FORMAT = 'png'
FIG_SIZE_STANDARD = (10, 6)
FIG_SIZE_WIDE = (14, 6)
FIG_SIZE_TALL = (10, 8)

# Color schemes
COLOR_MODELS = {
    'base': '#1f77b4',
    'base_a': '#ff7f0e',
    'ft': '#2ca02c',
    'ft_a': '#d62728'
}

COLOR_SUPERSECTORS = {
    'Financials': '#1f77b4',
    'Technology': '#ff7f0e',
    'Consumer & Retail': '#2ca02c',
    'Healthcare': '#d62728',
    'Industrials': '#9467bd',
    'Energy & Utilities': '#8c564b'
}

# =============================================================================
# COLUMN NAME MAPPINGS
# =============================================================================

# Score columns for each model (database uses 'con' for context/anchored)
SCORE_COLUMNS = {
    'base': 'base_score',
    'base_a': 'base_con_score',
    'ft': 'ft_score',
    'ft_a': 'ft_con_score'
}

# Certainty columns for each model
CERT_COLUMNS = {
    'base': 'base_cert',
    'base_a': 'base_con_cert',
    'ft': 'ft_cert',
    'ft_a': 'ft_con_cert'
}

# Rolling average column name template
def get_rol_avg_col(model, roltype):
    """Generate rolling average column name"""
    return f"{model}_rol_avg_{roltype}"

# Error metric column name templates
# Updated to match gemini_esg_data.db naming convention
def get_miss_col(model, roltype):
    """Generate miss (signed error) column name"""
    return f"{model}_{roltype}_miss"

def get_miss_abs_col(model, roltype):
    """Generate absolute miss column name"""
    return f"{model}_{roltype}_miss_absolute"

def get_dir_col(model, roltype=None):
    """Generate direction column name (no roltype variant)"""
    return f"{model}_dir"

def get_err_type_col(model, roltype=None):
    """Generate error type column name (no roltype variant)"""
    return f"{model}_err_type"

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Stop execution if missing data found
STOP_ON_MISSING_DATA = True

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-6
