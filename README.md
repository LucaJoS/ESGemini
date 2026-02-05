# ESGemini: Can LLMs Outpace the Market, Allowing for Information Arbitrage?

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the replication code for "ESGemini: Can LLMs Outpace the Market, Allowing for Information Arbitrage?"

We test whether Refinitiv/LSEG ESG scores can be anticipated from public corporate disclosures using Google's Gemini 1.5 Flash (base and fine-tuned variants). The study covers S&P 100 firms from 2014–2024 and evaluates four hypotheses:

- **H1**: LLMs can replicate official ESG scores with high fidelity
- **H2**: Fine-tuning improves cross-sectional ranking ability  
- **H3**: Model certainty correlates with forecast accuracy
- **H4**: Anticipatory signals generate exploitable returns

## Repository Structure

```
ESGemini/
├── data/
│   ├── gemini_esg_data.db                 # Main SQLite database (proprietary - not included)
│   ├── F-F_Research_Data_5_Factors_2x3_daily.xlsx  # Fama-French factors (included)
│   └── proprietary___not_available/       # Proprietary data (see Data Requirements)
│
├── scripts/
│   ├── config.py                          # Configuration and paths
│   ├── utils.py                           # Shared utility functions
│   ├── figure_style.py                    # Plot styling
│   ├── matched_CRSP_ticker.py             # CRSP ticker mapping
│   ├── RUN_ALL_ANALYSES.py                # Master execution script
│   │
│   ├── analysis/                          # Analysis scripts (15 total)
│   │   ├── 1_data_preparation.py          # Load and prepare data
│   │   ├── 2_h1_replication_fidelity.py   # H1: Replication analysis
│   │   ├── 3_h2_finetuning_value.py       # H2: Fine-tuning comparison
│   │   ├── 4_h3_structured_residuals.py   # H3: Error decomposition
│   │   ├── 5_h4_economic_value.py         # H4: Event study (main)
│   │   ├── 5b_h4_sector_neutral.py        # H4: Sector-neutral variant
│   │   ├── 5c_h4_ff_adjusted.py           # H4: Fama-French adjusted
│   │   ├── 5d_h4_sector_neutral_tertiles.py  # H4: Tertile sorting
│   │   ├── 5e_h4_in_vs_out_robustness.py  # H4: In-sample vs out-of-sample
│   │   ├── 6_enhancements.py              # Additional diagnostics
│   │   ├── 8_create_visualizations.py     # Generate all figures
│   │   ├── 9_robustness_market_cap.py     # Market cap robustness
│   │   ├── 10_robustness_transaction_costs.py  # Transaction cost sensitivity
│   │   ├── 11_bootstrap_inference.py      # Bootstrap inference
│   │   └── 12_calendar_time_portfolio.py  # Calendar-time analysis
│   │
│   └── setup/                             # Pre-inference methodology (documentation only)
│       ├── README.md
│       ├── 1_data_post_processing.py
│       ├── 2_create_joined_reports.py
│       └── 3_create_training_data.py
│
├── results/                               # Generated outputs (not tracked in git)
│   ├── csv/                               # Output tables
│   ├── figures/                           # Output figures (PDF & PNG)
│   ├── latex/                             # LaTeX table outputs
│   └── terminal_output.md                 # Full run log
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Script-to-Paper Mapping

All analysis scripts are in `scripts/analysis/`:

| Script | Paper Section | Output |
|--------|---------------|--------|
| `1_data_preparation.py` | - | Creates `working_data.db` |
| `2_h1_replication_fidelity.py` | Section 6.1 | Table 5, h1_results.csv |
| `3_h2_finetuning_value.py` | Section 6.2 | Table B2, h2_statistical_tests.csv |
| `4_h3_structured_residuals.py` | Section 7.2 | Table 6, h3_regression_results.csv |
| `5_h4_economic_value.py` | Section 7.1 | Table 7, h4_portfolio_returns.csv |
| `5b_h4_sector_neutral.py` | Section 7.3 | h4b_sector_neutral_returns.csv |
| `5c_h4_ff_adjusted.py` | Section 7.3 | h4c_ff_adjusted_returns.csv |
| `5d_h4_sector_neutral_tertiles.py` | Section 7.3 | h4d_tertiles.csv |
| `5e_h4_in_vs_out_robustness.py` | Section 7.3 | h4_in_vs_out_sample.csv |
| `6_enhancements.py` | Section 6.3 | enhancements_*.csv |
| `8_create_visualizations.py` | All | Figures 1-6 |
| `9_robustness_market_cap.py` | Section 7.4 | Figure 7, market_cap.csv |
| `10_robustness_transaction_costs.py` | Section 7.5 | Table 11, Figure 8 |
| `11_bootstrap_inference.py` | Section 7.6 | Table 9, bootstrap_inference.csv |
| `12_calendar_time_portfolio.py` | Section 7.7 | Table 10, calendar_time.csv |

Pre-inference methodology scripts are in `scripts/setup/` (documentation only, not runnable).

## Requirements

**Python version**: 3.9+

**Install dependencies**:
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn openpyxl
```

## Data Requirements

### Data Availability Statement

This research uses a combination of proprietary and publicly available data. Due to licensing restrictions from data providers, certain datasets cannot be redistributed. Below we detail what is included, what must be obtained separately, and how to configure the repository.

### Included Data (Shareable)
| File | Description | License |
|------|-------------|---------|
| `F-F_Research_Data_5_Factors_2x3_daily.xlsx` | Fama-French 5-factor daily returns | Kenneth R. French Data Library (public) |

### Partially Included Data
| File | What's Included | What's Not Included | Why |
|------|-----------------|---------------------|-----|
| `gemini_esg_data.db` | **Aggregate predictions** (rolling averages, miss metrics) and **sample flags** | Individual document-level predictions, certainty scores, raw LLM outputs | Refinitiv/LSEG ESG data licensing prohibits redistribution of underlying scores |

**Note on LLM Predictions**: While our Gemini model predictions are original work, they were generated from Refinitiv ESG scores as training/anchoring data. The rolling averages and aggregate metrics we include do not expose individual scores, allowing researchers to replicate the economic analyses (H1, H2, H4) without the raw proprietary data.

### Proprietary Data (Not Included - Must Be Obtained)
The following datasets are required for full replication:

| File | Source | Required For | How to Obtain |
|------|--------|--------------|---------------|
| `CRSP_SP100_ReturnData.db` | CRSP | H4 event studies, Bootstrap | WRDS subscription |
| `S&P100_price_hard.sqlite` | Refinitiv | Price data | Refinitiv Workspace/Eikon |
| `S&P100_sectors_hard.sqlite` | Refinitiv | Sector classification | Refinitiv Workspace/Eikon |
| `S&P100_volume_hard.sqlite` | Refinitiv | Trading volume | Refinitiv Workspace/Eikon |
| `S&P100_market_cap_hard.sqlite` | Refinitiv | Market capitalization | Refinitiv Workspace/Eikon |

**Setup**: Place proprietary data files in `data/proprietary___not_available/` or update paths in `scripts/config.py`.

### Replication Without Proprietary Data
Researchers without access to CRSP/Refinitiv can still:
- Replicate H1/H2/H3 analyses using the included aggregate metrics
- Verify statistical methodology and code logic
- Substitute with alternative return data (Yahoo Finance, etc.) by modifying `utils.py`

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/lucaschmidt/ESGemini-Replication.git
cd ESGemini-Replication
```

2. Install dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels scikit-learn openpyxl
```

3. (Optional) If you have proprietary CRSP/Refinitiv data, place it in `data/proprietary___not_available/`

4. Run all analyses:
```bash
cd scripts
python RUN_ALL_ANALYSES.py
```

**Note**: Scripts 5x, 11, and 12 require CRSP data. Other analyses will run with included data.

## Key Results

| Hypothesis | Finding |
|------------|---------|
| H1 | High replication fidelity with anchoring (R² = 0.81, MAE = 4.34 for base_a) |
| H2 | Fine-tuning improves ranking (Spearman ρ: 0.16 → 0.41 for ft) |
| H3 | Base certainty miscalibrated; fine-tuned certainty better calibrated |
| H4 | +0.64% net returns (p = 0.052 for base_a t-3/t+5), marginal bootstrap significance (p = 0.079) |

## Sample Information

- **Primary Sample**: 720 unique rating change events (2017–2024)
- **Out-of-Sample**: 118 events in 2024 (post-model knowledge cutoff)
- **Models Tested**: Base (zero-shot), Base Anchored, Fine-tuned, Fine-tuned Anchored

## Citation

```bibtex
@article{Schmidt2026,
  title={ESGemini: Can LLMs Outpace the Market, Allowing for Information Arbitrage?},
  author={Schmidt, Luca Johann},
  journal={Working Paper},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Version Notes

### Repository vs Paper Appendix
The paper appendix states that "model predictions" are available at this repository.
Due to Refinitiv/LSEG licensing restrictions, individual document-level predictions
cannot be redistributed. This repository contains:
- Complete analysis code for all hypotheses
- Methodology documentation
- Aggregate metrics sufficient for replication of economic analyses

Individual ESG scores and predictions require separate licensing from Refinitiv.

### Differences from Conference Submissions
This repository reflects minor technical corrections identified during code review
for public release. Specifically, certain robustness analyses contained minor
implementation errors that have been corrected. These corrections:
- Do not affect the paper's core findings (H1-H4 main results unchanged)
- Result in different specific values in some robustness tables
- Represent the authoritative, corrected analysis

The main economic finding (+0.64% net returns, p = 0.052) remains unchanged.
Should the paper be accepted for presentation, the corrected version will be presented.

## Contact

Luca Johann Schmidt

---

*Last updated: February 2026*

