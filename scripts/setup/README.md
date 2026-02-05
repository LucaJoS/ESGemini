# Setup Scripts (Pre-Inference Phase)

These scripts document the data preparation methodology performed before LLM inference.
They are provided for transparency and reproducibility documentation, not as runnable code.

## Scripts Included

| Script | Purpose |
|--------|---------|
| `1_data_post_processing.py` | Computes rolling averages, identifies ESG rating changes, calculates forecast errors |
| `2_create_joined_reports.py` | Concatenates individual corporate reports into period-level documents |
| `3_create_training_data.py` | Creates JSONL format for Google Gemini fine-tuning |

## Purpose

These scripts illustrate:
- How rolling average predictions are computed from daily LLM outputs
- How ESG rating change events are identified in Refinitiv data
- How documents are concatenated per fiscal period for inference
- The exact format used for Gemini fine-tuning training data

## Scripts Not Included

Some scripts from the original research pipeline are not included in this repository.
These typically involve proprietary API credentials, cloud infrastructure configurations,
or intermediate processing steps that depend on data not available for public distribution.

The complete methodology is documented in the paper (Section 4: Data and Methodology).

## Data Requirements

These scripts reference data that is not publicly available:
- Refinitiv ESG scores database
- CRSP daily stock returns
- Individual corporate disclosure documents

See the main `data/` README for details on data availability and licensing.
