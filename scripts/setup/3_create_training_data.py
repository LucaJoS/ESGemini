"""
CREATE TRAINING DATA: JSONL Format for Gemini Fine-Tuning
=========================================================

This script documents the methodology for creating training and validation
data for Google Gemini fine-tuning. It is provided for transparency.

METHODOLOGY OVERVIEW
--------------------
Fine-tuning requires supervised examples in JSONL format where each line
contains a conversation with user input (ESG report text) and model output
(ESG score prediction).

TRAINING/VALIDATION SPLIT
-------------------------
Based on rating period END dates:
- Training: Periods ending on or before 2023-01-01
- Validation: Periods ending between 2023-01-01 and 2024-01-01
- Test (held out): Periods ending after 2024-01-01

This temporal split ensures no look-ahead bias: the model is trained on
historical periods and validated/tested on future periods.

JSONL FORMAT
------------
Each line is a JSON object with this structure:
{
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "<PROMPT> + <REPORT_CONTENT>"}]
        },
        {
            "role": "model", 
            "parts": [{"text": "ESG_Score: XX.XX, Certainty: 1.00"}]
        }
    ]
}

The prompt instructs the model to act as an ESG analyst and predict the
next ESG score based on the provided corporate reports.

DATA REQUIREMENTS
-----------------
- gemini_esg_data.db: Database with actual_esg_score_matched column
- Joined report files: Period-level concatenated documents
- Period boundaries: From days_to_next_release column

NOTE: This script is not runnable without the proprietary source data.
"""

import sqlite3
import json
import os
import pandas as pd
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Database with ESG scores and period information
DB_PATH = '<DATA_PATH>/gemini_esg_data.db'

# Folder containing joined period reports
JOINED_REPORTS_FOLDER = '<DATA_PATH>/joined_reports/'

# Output folder for JSONL training files
OUTPUT_FOLDER = '<DATA_PATH>/training_data/'

# Output filenames
TRAINING_FILE = 'SP100_training.jsonl'
VALIDATION_FILE = 'SP100_validation.jsonl'

# S&P 100 tickers
SP100_TICKERS = [
    'AAPL.OQ', 'ABBV.N', 'ABT.N', 'ACN.N', 'ADBE.OQ', 'AIG.N', 'AMD.OQ',
    'AMGN.OQ', 'AMT.N', 'AMZN.OQ', 'AVGO.OQ', 'AXP.N', 'BA.N', 'BAC.N',
    # ... (full list of 100 tickers)
]

# The exact prompt used for fine-tuning
FIXED_PROMPT = """You are an expert ESG (Environmental, Social, and Governance) financial analyst. Your task is to analyze comprehensive, chronologically-ordered corporate reports for a specific company over a given timeframe. Based *only* on the provided text, you will predict the company's next ESG Score that will be assigned *after* this period ends.


You must provide two things:
1. Your prediction for the new ESG Score.
2. A certainty score from 0.0 to 1.0, representing your confidence in the prediction. A score of 1.0 means you are absolutely certain, and a score of 0.0 means you are completely guessing. Base your certainty on the quality and relevance of the information in the reports.


Your output must be in this exact format:
ESG_Score: XX.XX, Certainty: X.XX


"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_jsonl_entry(ticker, report_content, esg_score):
    """
    Create a single JSONL training entry.
    
    The entry follows Google's Gemini fine-tuning format with a
    user message (prompt + report) and model response (score + certainty).
    
    For training data, certainty is set to 1.00 as we are providing
    the ground truth ESG score.
    
    Args:
        ticker: Instrument identifier (for reference, not used in prompt)
        report_content: Full text of concatenated corporate reports
        esg_score: Actual ESG score (ground truth label)
        
    Returns:
        dict: JSONL-compatible training entry
    """
    entry = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": FIXED_PROMPT + report_content + "\n\n"}]
            },
            {
                "role": "model",
                "parts": [{"text": f"ESG_Score: {esg_score:.2f}, Certainty: 1.00"}]
            }
        ]
    }
    return entry


# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================

def create_training_validation_data(
    db_path, 
    joined_reports_folder, 
    output_folder, 
    tickers,
    train_cutoff=datetime(2023, 1, 1),
    validation_cutoff=datetime(2024, 1, 1)
):
    """
    Create JSONL training and validation files for Gemini fine-tuning.
    
    The temporal split is based on period END dates to prevent look-ahead bias:
    - Training: Historical periods (end <= train_cutoff)
    - Validation: Recent periods (train_cutoff < end <= validation_cutoff)
    - Test: Future periods (end > validation_cutoff) - excluded from fine-tuning
    
    Args:
        db_path: Path to database with ESG scores
        joined_reports_folder: Directory with period-level report files
        output_folder: Directory to save JSONL files
        tickers: List of instruments to process
        train_cutoff: End date for training periods
        validation_cutoff: End date for validation periods
    """
    os.makedirs(output_folder, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    all_training_data = []
    all_validation_data = []
    
    stats = {
        'total_periods': 0,
        'training_periods': 0,
        'validation_periods': 0,
        'skipped_no_score': 0,
        'skipped_no_file': 0
    }
    
    for instrument in tickers:
        # Load periods and their ESG scores
        df = pd.read_sql_query(f"""
            SELECT date, days_to_next_release, actual_esg_score_matched 
            FROM master_predictions 
            WHERE instrument = '{instrument}' 
            ORDER BY date
        """, conn)
        
        if df.empty:
            continue
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Identify periods and their next ESG scores
        periods_with_scores = []
        current_period_start = None
        
        for idx, row in df.iterrows():
            if current_period_start is None:
                current_period_start = row['date']
            
            if row['days_to_next_release'] == 0:
                # Look ahead for next period's ESG score
                if idx + 1 < len(df):
                    next_esg_score = df.iloc[idx + 1]['actual_esg_score_matched']
                    periods_with_scores.append({
                        'start_date': current_period_start,
                        'end_date': row['date'],
                        'next_esg_score': next_esg_score
                    })
                current_period_start = None
        
        stats['total_periods'] += len(periods_with_scores)
        
        # Process each period
        for period in periods_with_scores:
            start_date = period['start_date']
            end_date = period['end_date']
            next_esg_score = period['next_esg_score']
            
            # Skip if no ESG score available
            if pd.isna(next_esg_score):
                stats['skipped_no_score'] += 1
                continue
            
            # Find corresponding joined report file
            expected_filename = f"JF_{instrument}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.txt"
            filepath = os.path.join(joined_reports_folder, expected_filename)
            
            # Check if file exists and is not empty
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                stats['skipped_no_file'] += 1
                continue
            
            # Read report content
            with open(filepath, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Assign to training or validation based on END date
            entry_data = {
                'ticker': instrument,
                'start_date': start_date,
                'end_date': end_date,
                'esg_score': next_esg_score,
                'content': report_content
            }
            
            if end_date <= train_cutoff:
                all_training_data.append(entry_data)
                stats['training_periods'] += 1
            elif end_date <= validation_cutoff:
                all_validation_data.append(entry_data)
                stats['validation_periods'] += 1
            # Periods after validation_cutoff are excluded (test set)
    
    conn.close()
    
    # Write JSONL files
    training_path = os.path.join(output_folder, TRAINING_FILE)
    with open(training_path, 'w', encoding='utf-8') as f:
        for data in all_training_data:
            entry = create_jsonl_entry(data['ticker'], data['content'], data['esg_score'])
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    validation_path = os.path.join(output_folder, VALIDATION_FILE)
    with open(validation_path, 'w', encoding='utf-8') as f:
        for data in all_validation_data:
            entry = create_jsonl_entry(data['ticker'], data['content'], data['esg_score'])
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nData creation complete:")
    print(f"  Total periods found: {stats['total_periods']}")
    print(f"  Training examples: {stats['training_periods']}")
    print(f"  Validation examples: {stats['validation_periods']}")
    print(f"  Skipped (no ESG score): {stats['skipped_no_score']}")
    print(f"  Skipped (no/empty file): {stats['skipped_no_file']}")
    print(f"\nOutput files:")
    print(f"  {training_path}")
    print(f"  {validation_path}")


# =============================================================================
# MAIN EXECUTION (for reference only)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CREATE TRAINING DATA SCRIPT")
    print("=" * 70)
    print("\nNOTE: This script requires proprietary data not included in the repository.")
    print("It is provided for methodology documentation only.\n")
    
    print("Process that would be executed:")
    print("  1. Load period information and ESG scores from database")
    print("  2. Match periods to joined report files")
    print("  3. Split temporally: training <= 2023, validation 2023-2024")
    print("  4. Create JSONL entries with prompt + report content")
    print("  5. Write to SP100_training.jsonl and SP100_validation.jsonl")
    print("\nThe JSONL format follows Google Gemini fine-tuning specifications.")
