"""
CREATE JOINED REPORTS: Period-Level Document Concatenation
==========================================================

This script documents the methodology for creating period-level input documents
for LLM inference. It is provided for transparency; the source data is proprietary.

METHODOLOGY OVERVIEW
--------------------
For each ESG rating period (time between consecutive rating changes):

1. IDENTIFY PERIOD BOUNDARIES
   - Use days_to_next_release = 0 to mark rating change events
   - Each period spans from after one rating change to the next

2. COLLECT RELEVANT DOCUMENTS
   - Find all corporate disclosure documents filed during the period
   - Documents include: SEC filings, earnings transcripts, press releases
   - Filter to those matching the instrument/ticker

3. CONCATENATE CHRONOLOGICALLY
   - Sort documents by filing date (ascending)
   - Join with double newlines as separator
   - Save as single text file per period

OUTPUT FORMAT
-------------
Files named: JF_{TICKER}_{START_DATE}_{END_DATE}.txt
Example: JF_AAPL.OQ_2022-01-03_2022-12-31.txt

These concatenated documents serve as input to the Gemini model for
ESG score prediction. The chronological ordering preserves the temporal
flow of information disclosure.

DATA REQUIREMENTS
-----------------
- gemini_esg_data.db: Database with rating change periods identified
- Individual report TXT files: Corporate disclosures in text format
- Period boundaries: From days_to_next_release column

NOTE: This script is not runnable without the proprietary source data.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION - Replace with your paths
# =============================================================================

# Database with period information
DB_PATH = '<DATA_PATH>/gemini_esg_data.db'

# Folder containing individual report text files
TXT_FOLDER = '<DATA_PATH>/corporate_reports_txt/'

# Output folder for joined reports
OUTPUT_FOLDER = '<DATA_PATH>/joined_reports/'

# S&P 100 tickers
SP100_TICKERS = [
    'AAPL.OQ', 'ABBV.N', 'ABT.N', 'ACN.N', 'ADBE.OQ', 'AIG.N', 'AMD.OQ',
    'AMGN.OQ', 'AMT.N', 'AMZN.OQ', 'AVGO.OQ', 'AXP.N', 'BA.N', 'BAC.N',
    # ... (full list of 100 tickers)
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_info_from_filename(filename):
    """
    Parse document filename to extract filing date and ticker.
    
    Expected format: "Filing Date - YYYY-MMM-DD - TICKER - Description.txt"
    Example: "Filing Date - 2022-Mar-15 - AAPL.OQ - Q1 Earnings Call.txt"
    
    Returns:
        tuple: (datetime, ticker) or (None, None) if parsing fails
    """
    try:
        parts = filename.split(' - ')
        if len(parts) >= 3 and parts[0] == 'Filing Date':
            date = datetime.strptime(parts[1], '%Y-%b-%d')
            ticker = parts[2]
            return date, ticker
    except (ValueError, IndexError):
        pass
    return None, None


def get_size_category(size_bytes):
    """Categorize file size for statistics."""
    size_mb = size_bytes / (1024 * 1024)
    size_kb = size_bytes / 1024
    
    if size_mb > 10: return '>10MB'
    elif size_mb > 5: return '>5MB'
    elif size_mb > 1: return '>1MB'
    elif size_kb > 500: return '>500KB'
    elif size_kb > 100: return '>100KB'
    elif size_kb > 50: return '>50KB'
    else: return '<50KB'


# =============================================================================
# MAIN PROCESSING LOGIC
# =============================================================================

def create_joined_reports(db_path, txt_folder, output_folder, tickers):
    """
    Create period-level joined documents for LLM inference.
    
    Process:
    1. For each instrument, load period boundaries from database
    2. Collect all text files falling within each period
    3. Concatenate and save as single file per period
    
    Args:
        db_path: Path to SQLite database with period information
        txt_folder: Directory containing individual report TXT files
        output_folder: Directory to save joined reports
        tickers: List of instrument tickers to process
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Build index of available text files
    txt_files = {}  # {full_path: (date, ticker)}
    for root, dirs, files in os.walk(txt_folder):
        for filename in files:
            if filename.endswith('.txt'):
                file_date, file_ticker = extract_info_from_filename(filename)
                if file_date and file_ticker:
                    full_path = os.path.join(root, filename)
                    txt_files[full_path] = (file_date, file_ticker)
    
    print(f"Found {len(txt_files)} text files with valid metadata")
    
    # Process each instrument
    conn = sqlite3.connect(db_path)
    stats = {'total_periods': 0, 'files_created': 0, 'total_size_mb': 0}
    
    for instrument in tickers:
        # Load period boundaries
        df = pd.read_sql_query(f"""
            SELECT date, days_to_next_release 
            FROM master_predictions 
            WHERE instrument = '{instrument}' 
            ORDER BY date
        """, conn)
        
        if df.empty:
            continue
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Identify periods (each ends when days_to_next_release = 0)
        periods = []
        current_period = []
        for _, row in df.iterrows():
            current_period.append(row['date'])
            if row['days_to_next_release'] == 0:
                periods.append((min(current_period), max(current_period)))
                current_period = []
        if current_period:
            periods.append((min(current_period), max(current_period)))
        
        stats['total_periods'] += len(periods)
        
        # Process each period
        for start_date, end_date in periods:
            concatenated_text = []
            
            # Find files matching this period and ticker
            for filepath, (file_date, file_ticker) in sorted(
                txt_files.items(), key=lambda x: x[1][0]
            ):
                if file_ticker == instrument and start_date <= file_date <= end_date:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        concatenated_text.append(f.read())
            
            # Write joined output
            output_filename = f"JF_{instrument}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.txt"
            output_path = os.path.join(output_folder, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(concatenated_text))
            
            file_size = os.path.getsize(output_path)
            stats['files_created'] += 1
            stats['total_size_mb'] += file_size / (1024 * 1024)
    
    conn.close()
    
    print(f"\nProcessing complete:")
    print(f"  Total periods: {stats['total_periods']}")
    print(f"  Files created: {stats['files_created']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")


# =============================================================================
# MAIN EXECUTION (for reference only)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CREATE JOINED REPORTS SCRIPT")
    print("=" * 70)
    print("\nNOTE: This script requires proprietary data not included in the repository.")
    print("It is provided for methodology documentation only.\n")
    
    print("Process that would be executed:")
    print("  1. Index all available corporate disclosure text files")
    print("  2. Identify ESG rating periods from database")
    print("  3. For each period, concatenate relevant documents chronologically")
    print("  4. Save as single text file per period for LLM input")
    print("\nOutput format: JF_{TICKER}_{START}_{END}.txt")
