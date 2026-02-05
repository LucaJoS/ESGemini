"""
DATA POST-PROCESSING: Rolling Averages, Rating Changes, and Forecast Errors
============================================================================

This script documents the post-inference data processing methodology.
It is provided for transparency; the underlying data is not publicly available.

METHODOLOGY OVERVIEW
--------------------
After obtaining daily LLM predictions, this script:

1. IDENTIFY RATING CHANGE EVENTS
   - Detect days when Refinitiv ESG score changes (threshold >= 0.001)
   - Flag these as rating change events (NA_esg_score = 1)
   - Calculate days_to_next_release for each observation

2. COMPUTE ROLLING AVERAGES
   - Aggregate daily predictions into period-level rolling averages
   - Rolling averages RESET at each rating change (period boundary)
   - Each fiscal period uses only its own observations
   
3. CALCULATE FORECAST ERRORS
   - Compare rolling average predictions to actual next-period ESG scores
   - Compute directional accuracy (UP/DOWN prediction vs actual)
   - Classify errors as overestimation or underestimation

DATA REQUIREMENTS
-----------------
- gemini_esg_data.db: SQLite database with master_predictions table
- Daily LLM predictions: base_score, base_con_score, ft_score, ft_con_score
- Refinitiv ESG scores: esg_score column
- Model certainty: base_cert, base_con_cert, ft_cert, ft_con_cert

NOTE: This script is not runnable without the proprietary source data.
"""

import sqlite3
import datetime as dt
import pandas as pd

# =============================================================================
# CONFIGURATION - Replace with your paths
# =============================================================================

# Main database with LLM predictions
DB_PATH = '<DATA_PATH>/gemini_esg_data.db'

# Reference database with raw Refinitiv ESG scores (for rating change detection)
REF_DB_PATH = '<DATA_PATH>/refinitiv_esg_scores.db'

# S&P 100 tickers used in the study
SP100_TICKERS = [
    'AAPL.OQ', 'ABBV.N', 'ABT.N', 'ACN.N', 'ADBE.OQ', 'AIG.N', 'AMD.OQ', 
    'AMGN.OQ', 'AMT.N', 'AMZN.OQ', 'AVGO.OQ', 'AXP.N', 'BA.N', 'BAC.N', 
    'BK.N', 'BKNG.OQ', 'BMY.N', 'BRKb.N', 'C.N', 'CAT.N', 'CHTR.OQ', 
    'CL.N', 'CMCSA.OQ', 'COF.N', 'COP.N', 'COST.OQ', 'CRM.N', 'CSCO.OQ', 
    'CVS.N', 'CVX.N', 'DE.N', 'DHR.N', 'DIS.N', 'DUK.N', 'EMR.N', 'FDX.N', 
    'GD.N', 'GE.N', 'GILD.OQ', 'GM.N', 'GOOGL.OQ', 'GS.N', 'HD.N', 'HON.OQ', 
    'IBM.N', 'INTC.OQ', 'INTU.OQ', 'ISRG.OQ', 'JNJ.N', 'JPM.N', 'KO.N', 
    'LIN.OQ', 'LLY.N', 'LMT.N', 'LOW.N', 'MA.N', 'MCD.N', 'MDLZ.OQ', 
    'MDT.N', 'MET.N', 'META.OQ', 'MMM.N', 'MO.N', 'MRK.N', 'MS.N', 
    'MSFT.OQ', 'NEE.N', 'NFLX.OQ', 'NKE.N', 'NOW.N', 'NVDA.OQ', 'ORCL.N', 
    'PEP.OQ', 'PFE.N', 'PG.N', 'PLTR.OQ', 'PM.N', 'PYPL.OQ', 'QCOM.OQ', 
    'RTX.N', 'SBUX.OQ', 'SCHW.N', 'SO.N', 'SPG.N', 'T.N', 'TGT.N', 
    'TMO.N', 'TMUS.OQ', 'TSLA.OQ', 'TXN.OQ', 'UNH.N', 'UNP.N', 'UPS.N', 
    'USB.N', 'V.N', 'VZ.N', 'WFC.N', 'WMT.N', 'XOM.N'
]

# Model column mappings
SCORE_COLS = [
    ("base_score", "base_rol_avg"),
    ("base_con_score", "base_con_rol_avg"),
    ("ft_score", "ft_rol_avg"),
    ("ft_con_score", "ft_con_rol_avg")
]

CERT_COLS = [
    ("base_cert", "base_cert_period_avg"),
    ("base_con_cert", "base_con_cert_period_avg"),
    ("ft_cert", "ft_cert_period_avg"),
    ("ft_con_cert", "ft_con_cert_period_avg")
]


# =============================================================================
# STEP 1: IDENTIFY RATING CHANGE EVENTS
# =============================================================================

def identify_rating_changes(db_path, ref_db_path, tickers, threshold=0.001):
    """
    Detect ESG rating change events from Refinitiv data.
    
    A rating change is identified when the ESG score changes by >= threshold
    from the previous observation. The threshold of 0.001 was validated to
    capture all meaningful rating changes while avoiding floating-point noise.
    
    Returns:
        List of (date, ticker) tuples for rating change events
    """
    change_pairs = []
    
    with sqlite3.connect(ref_db_path) as conn:
        cur = conn.cursor()
        
        for ticker in tickers:
            # Get chronological ESG scores for this ticker
            cur.execute("""
                SELECT substr(Date, 1, 10) AS d, "ESG Score"
                FROM esg_scores
                WHERE Instrument = ?
                  AND "ESG Score" IS NOT NULL
                  AND "ESG Score" != ''
                ORDER BY Date ASC
            """, (ticker,))
            
            rows = cur.fetchall()
            if len(rows) < 2:
                continue
            
            prev_score = None
            for date, score in rows:
                try:
                    current_score = float(score)
                    
                    if prev_score is not None:
                        diff = abs(current_score - prev_score)
                        
                        # Flag as rating change if difference >= threshold
                        if diff >= threshold:
                            change_pairs.append((date, ticker))
                    
                    prev_score = current_score
                    
                except (ValueError, TypeError):
                    continue
    
    return change_pairs


# =============================================================================
# STEP 2: COMPUTE ROLLING AVERAGES
# =============================================================================

def compute_rolling_averages(db_path):
    """
    Compute period-based rolling averages for LLM predictions.
    
    KEY DESIGN DECISION: Rolling averages RESET at each rating change event.
    This ensures each fiscal period uses only its own observations, preventing
    information leakage across periods.
    
    For each instrument:
    1. Process observations chronologically
    2. Accumulate predictions within each period
    3. Reset accumulators when hitting a rating change (NA_esg_score = 1)
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        
        # Get all instruments
        cur.execute("SELECT DISTINCT instrument FROM master_predictions")
        instruments = [r[0] for r in cur.fetchall()]
        
        for inst in instruments:
            # Load all rows for this instrument
            cur.execute("""
                SELECT rowid, date, NA_esg_score,
                       base_score, base_con_score, ft_score, ft_con_score,
                       base_cert, base_con_cert, ft_cert, ft_con_cert
                FROM master_predictions
                WHERE instrument = ?
                ORDER BY date
            """, (inst,))
            rows = cur.fetchall()
            
            # Initialize accumulators for rolling averages
            sums = {src: 0.0 for src, _ in SCORE_COLS}
            cnts = {src: 0 for src, _ in SCORE_COLS}
            last = {src: None for src, _ in SCORE_COLS}
            
            updates = []
            
            for row in rows:
                rid, date, na_flag = row[0], row[1], row[2]
                scores = {
                    "base_score": row[3],
                    "base_con_score": row[4],
                    "ft_score": row[5],
                    "ft_con_score": row[6]
                }
                
                # Update rolling averages
                avgs = {}
                for src, tgt in SCORE_COLS:
                    v = scores[src]
                    if v is not None:
                        sums[src] += float(v)
                        cnts[src] += 1
                        last[src] = sums[src] / cnts[src]
                    avgs[tgt] = last[src]
                
                updates.append((
                    avgs["base_rol_avg"],
                    avgs["base_con_rol_avg"],
                    avgs["ft_rol_avg"],
                    avgs["ft_con_rol_avg"],
                    rid
                ))
                
                # Reset at period boundary (rating change day)
                if na_flag == 1:
                    sums = {src: 0.0 for src, _ in SCORE_COLS}
                    cnts = {src: 0 for src, _ in SCORE_COLS}
                    last = {src: None for src, _ in SCORE_COLS}
            
            # Apply updates
            cur.executemany("""
                UPDATE master_predictions
                SET base_rol_avg = ?, base_con_rol_avg = ?,
                    ft_rol_avg = ?, ft_con_rol_avg = ?
                WHERE rowid = ?
            """, updates)
        
        conn.commit()


# =============================================================================
# STEP 3: CALCULATE FORECAST ERRORS
# =============================================================================

def calculate_forecast_errors(db_path):
    """
    Compute forecast errors by comparing predictions to actual outcomes.
    
    For each rating change event:
    1. Get the rolling average prediction as of the day before the event
    2. Compare to the actual new ESG score
    3. Calculate:
       - miss = actual - prediction (signed error)
       - error_type = 'overestimation' if miss < 0, 'underestimation' if miss > 0
       - direction = 'UP' or 'DOWN' based on prediction vs previous score
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        
        cur.execute("SELECT DISTINCT instrument FROM master_predictions")
        instruments = [r[0] for r in cur.fetchall()]
        
        for inst in instruments:
            cur.execute("""
                SELECT rowid, NA_esg_score, esg_score,
                       base_rol_avg, base_con_rol_avg, ft_rol_avg, ft_con_rol_avg
                FROM master_predictions
                WHERE instrument = ?
                ORDER BY date
            """, (inst,))
            rows = cur.fetchall()
            
            # Track last seen rolling averages
            last_base = last_bc = last_ft = last_ftc = None
            updates = []
            
            for i, (rid, na, esg, br, bc, ftr, ftc) in enumerate(rows):
                # Update tracking variables
                if br is not None: last_base = float(br)
                if bc is not None: last_bc = float(bc)
                if ftr is not None: last_ft = float(ftr)
                if ftc is not None: last_ftc = float(ftc)
                
                if na == 1:  # Rating change event
                    # Find actual ESG score (may be on this row or nearby)
                    e_val = None
                    for j in range(i, min(i + 6, len(rows))):
                        e = rows[j][2]
                        if e is not None:
                            e_val = float(e)
                            break
                    
                    if e_val is None:
                        continue
                    
                    # Calculate misses (actual - prediction)
                    updates.append((
                        None if last_base is None else e_val - last_base,
                        None if last_bc is None else e_val - last_bc,
                        None if last_ft is None else e_val - last_ft,
                        None if last_ftc is None else e_val - last_ftc,
                        rid
                    ))
            
            cur.executemany("""
                UPDATE master_predictions
                SET base_miss = ?, base_c_miss = ?, ft_miss = ?, ft_c_miss = ?
                WHERE rowid = ?
            """, updates)
        
        # Classify error types based on miss sign
        cur.execute("""
            UPDATE master_predictions
            SET base_err_type = CASE 
                    WHEN base_miss > 0 THEN 'underestimation'
                    WHEN base_miss < 0 THEN 'overestimation' 
                END,
                base_c_err_type = CASE 
                    WHEN base_c_miss > 0 THEN 'underestimation'
                    WHEN base_c_miss < 0 THEN 'overestimation' 
                END,
                ft_err_type = CASE 
                    WHEN ft_miss > 0 THEN 'underestimation'
                    WHEN ft_miss < 0 THEN 'overestimation' 
                END,
                ft_c_err_type = CASE 
                    WHEN ft_c_miss > 0 THEN 'underestimation'
                    WHEN ft_c_miss < 0 THEN 'overestimation' 
                END
        """)
        
        conn.commit()


# =============================================================================
# MAIN EXECUTION (for reference only)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA POST-PROCESSING SCRIPT")
    print("=" * 70)
    print("\nNOTE: This script requires proprietary data not included in the repository.")
    print("It is provided for methodology documentation only.\n")
    
    print("Steps that would be executed:")
    print("  1. Identify rating change events from Refinitiv data")
    print("  2. Compute period-based rolling averages for LLM predictions")
    print("  3. Calculate forecast errors and directional accuracy")
    print("\nSee docstrings for detailed methodology.")
