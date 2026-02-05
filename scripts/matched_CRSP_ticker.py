"""
CRSP TICKER MAPPING FILE
========================
This file contains mappings from your dataset instruments to CRSP tickers.
Purpose: Handle ticker changes and naming differences between your data and CRSP

USAGE:
    from matched_CRSP_ticker import get_crsp_ticker
    crsp_ticker = get_crsp_ticker(instrument, date)
"""

from datetime import datetime
import pandas as pd

# =============================================================================
# STATIC TICKER MAPPINGS (simple name differences)
# =============================================================================
# Format: 'instrument_base_ticker': 'crsp_ticker'

STATIC_TICKER_MAP = {
    'BRKb': 'BH',  # Berkshire Hathaway uses 'BH' in CRSP, not 'BRK.B' or 'BRKb'
}

# =============================================================================
# TICKER CHANGE MAPPINGS (companies that changed tickers over time)
# =============================================================================
# Format: 'instrument_base_ticker': [(crsp_ticker, start_date, end_date), ...]
# Dates are inclusive. Use None for open-ended ranges.

TICKER_CHANGES = {
    # Facebook -> Meta (changed June 9, 2022)
    'META': [
        ('FB',   datetime(2013, 1, 1),  datetime(2022, 6, 8)),   # Before rename
        ('META', datetime(2022, 6, 9),  None),                   # After rename
    ],
    
    # Priceline -> Booking (changed Feb 27, 2018)
    'BKNG': [
        ('PCLN', datetime(2013, 1, 1),  datetime(2018, 2, 26)),  # Before rename
        ('BKNG', datetime(2018, 2, 27), None),                   # After rename
    ],
    
    # United Technologies -> Raytheon (merged April 3, 2020)
    'RTX': [
        ('UTX', datetime(2013, 1, 1),  datetime(2020, 4, 2)),   # Before merger
        ('RTX', datetime(2020, 4, 3),  None),                   # After merger
    ],
    
    # Google -> Alphabet (class C shares split April 3, 2014)
    # Note: Your data uses GOOGL.OQ which maps directly to GOOGL
    'GOOG': [
        ('GOOG',  datetime(2013, 1, 1),  datetime(2014, 4, 2)),  # Before split
        ('GOOGL', datetime(2014, 4, 3),  None),                  # After split (Class A)
    ],
    
    # Linde - has a data gap in CRSP
    # Old Linde AG: 2013-12-31 to 2014-12-22
    # New Linde plc (after Praxair merger): 2018-10-31 onwards
    # There is NO CRSP data for LIN from 2014-12-23 to 2018-10-30
    'LIN': [
        ('LIN', datetime(2013, 1, 1),  datetime(2014, 12, 22)),  # Old Linde AG
        # GAP: No data 2014-12-23 to 2018-10-30
        ('LIN', datetime(2018, 10, 31), None),                    # New Linde plc
    ],
}

# =============================================================================
# DATA GAPS (periods where CRSP has no data for a ticker)
# =============================================================================
# These are legitimate gaps where no CRSP return data exists

DATA_GAPS = {
    'LIN': [
        (datetime(2014, 12, 23), datetime(2018, 10, 30)),  # No LIN data during this period
    ],
}

# =============================================================================
# LOOKUP FUNCTION
# =============================================================================

def get_crsp_ticker(instrument, date):
    """
    Get the CRSP ticker for a given instrument and date.
    
    Args:
        instrument: str - Your instrument name (e.g., 'META.OQ', 'BRKb.N')
        date: datetime or str - The date for the lookup
        
    Returns:
        str or None - The CRSP ticker, or None if no mapping exists
    """
    # Convert date if string
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    
    # Extract base ticker (remove exchange suffix)
    base_ticker = instrument.split('.')[0] if '.' in instrument else instrument
    
    # Check static mapping first
    if base_ticker in STATIC_TICKER_MAP:
        return STATIC_TICKER_MAP[base_ticker]
    
    # Check ticker changes
    if base_ticker in TICKER_CHANGES:
        for crsp_ticker, start_date, end_date in TICKER_CHANGES[base_ticker]:
            if start_date <= date and (end_date is None or date <= end_date):
                return crsp_ticker
        # If no matching date range, return None (data gap)
        return None
    
    # Default: use base ticker as-is
    return base_ticker


def is_in_data_gap(instrument, date):
    """
    Check if a date falls within a known CRSP data gap.
    
    Args:
        instrument: str - Your instrument name
        date: datetime or str - The date to check
        
    Returns:
        bool - True if the date is in a data gap
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    elif hasattr(date, 'to_pydatetime'):
        date = date.to_pydatetime()
    
    base_ticker = instrument.split('.')[0] if '.' in instrument else instrument
    
    if base_ticker in DATA_GAPS:
        for gap_start, gap_end in DATA_GAPS[base_ticker]:
            if gap_start <= date <= gap_end:
                return True
    return False


# =============================================================================
# SUMMARY OF ALL MAPPINGS (for manual review)
# =============================================================================

ALL_INSTRUMENTS = [
    # Format: (instrument, base_ticker, crsp_ticker_or_rule, notes)
    ('AAPL.OQ', 'AAPL', 'AAPL', 'Direct match'),
    ('ABBV.N', 'ABBV', 'ABBV', 'Direct match'),
    ('ABT.N', 'ABT', 'ABT', 'Direct match'),
    ('ACN.N', 'ACN', 'ACN', 'Direct match'),
    ('ADBE.OQ', 'ADBE', 'ADBE', 'Direct match'),
    ('AIG.N', 'AIG', 'AIG', 'Direct match'),
    ('AMD.OQ', 'AMD', 'AMD', 'Direct match'),
    ('AMGN.OQ', 'AMGN', 'AMGN', 'Direct match'),
    ('AMT.N', 'AMT', 'AMT', 'Direct match'),
    ('AMZN.OQ', 'AMZN', 'AMZN', 'Direct match'),
    ('AVGO.OQ', 'AVGO', 'AVGO', 'Direct match'),
    ('AXP.N', 'AXP', 'AXP', 'Direct match'),
    ('BA.N', 'BA', 'BA', 'Direct match'),
    ('BAC.N', 'BAC', 'BAC', 'Direct match'),
    ('BK.N', 'BK', 'BK', 'Direct match'),
    ('BKNG.OQ', 'BKNG', 'PCLN->BKNG', 'Priceline renamed to Booking on 2018-02-27'),
    ('BMY.N', 'BMY', 'BMY', 'Direct match'),
    ('BRKb.N', 'BRKb', 'BH', 'Berkshire uses BH in CRSP'),
    ('C.N', 'C', 'C', 'Direct match'),
    ('CAT.N', 'CAT', 'CAT', 'Direct match'),
    ('CHTR.OQ', 'CHTR', 'CHTR', 'Direct match'),
    ('CL.N', 'CL', 'CL', 'Direct match'),
    ('CMCSA.OQ', 'CMCSA', 'CMCSA', 'Direct match'),
    ('COF.N', 'COF', 'COF', 'Direct match'),
    ('COP.N', 'COP', 'COP', 'Direct match'),
    ('COST.OQ', 'COST', 'COST', 'Direct match'),
    ('CRM.N', 'CRM', 'CRM', 'Direct match'),
    ('CSCO.OQ', 'CSCO', 'CSCO', 'Direct match'),
    ('CVS.N', 'CVS', 'CVS', 'Direct match'),
    ('CVX.N', 'CVX', 'CVX', 'Direct match'),
    ('DE.N', 'DE', 'DE', 'Direct match'),
    ('DHR.N', 'DHR', 'DHR', 'Direct match'),
    ('DIS.N', 'DIS', 'DIS', 'Direct match'),
    ('DUK.N', 'DUK', 'DUK', 'Direct match'),
    ('EMR.N', 'EMR', 'EMR', 'Direct match'),
    ('FDX.N', 'FDX', 'FDX', 'Direct match'),
    ('GD.N', 'GD', 'GD', 'Direct match'),
    ('GE.N', 'GE', 'GE', 'Direct match'),
    ('GILD.OQ', 'GILD', 'GILD', 'Direct match'),
    ('GM.N', 'GM', 'GM', 'Direct match'),
    ('GOOGL.OQ', 'GOOGL', 'GOOGL', 'Direct match (Class A shares)'),
    ('GS.N', 'GS', 'GS', 'Direct match'),
    ('HD.N', 'HD', 'HD', 'Direct match'),
    ('HON.OQ', 'HON', 'HON', 'Direct match'),
    ('IBM.N', 'IBM', 'IBM', 'Direct match'),
    ('INTC.OQ', 'INTC', 'INTC', 'Direct match'),
    ('INTU.OQ', 'INTU', 'INTU', 'Direct match'),
    ('ISRG.OQ', 'ISRG', 'ISRG', 'Direct match'),
    ('JNJ.N', 'JNJ', 'JNJ', 'Direct match'),
    ('JPM.N', 'JPM', 'JPM', 'Direct match'),
    ('KO.N', 'KO', 'KO', 'Direct match'),
    ('LIN.OQ', 'LIN', 'LIN (with gap)', 'Data gap: 2014-12-23 to 2018-10-30'),
    ('LLY.N', 'LLY', 'LLY', 'Direct match'),
    ('LMT.N', 'LMT', 'LMT', 'Direct match'),
    ('LOW.N', 'LOW', 'LOW', 'Direct match'),
    ('MA.N', 'MA', 'MA', 'Direct match'),
    ('MCD.N', 'MCD', 'MCD', 'Direct match'),
    ('MDLZ.OQ', 'MDLZ', 'MDLZ', 'Direct match'),
    ('MDT.N', 'MDT', 'MDT', 'Direct match'),
    ('MET.N', 'MET', 'MET', 'Direct match'),
    ('META.OQ', 'META', 'FB->META', 'Facebook renamed to Meta on 2022-06-09'),
    ('MMM.N', 'MMM', 'MMM', 'Direct match'),
    ('MO.N', 'MO', 'MO', 'Direct match'),
    ('MRK.N', 'MRK', 'MRK', 'Direct match'),
    ('MS.N', 'MS', 'MS', 'Direct match'),
    ('MSFT.OQ', 'MSFT', 'MSFT', 'Direct match'),
    ('NEE.N', 'NEE', 'NEE', 'Direct match'),
    ('NFLX.OQ', 'NFLX', 'NFLX', 'Direct match'),
    ('NKE.N', 'NKE', 'NKE', 'Direct match'),
    ('NOW.N', 'NOW', 'NOW', 'Direct match'),
    ('NVDA.OQ', 'NVDA', 'NVDA', 'Direct match'),
    ('ORCL.N', 'ORCL', 'ORCL', 'Direct match'),
    ('PEP.OQ', 'PEP', 'PEP', 'Direct match'),
    ('PFE.N', 'PFE', 'PFE', 'Direct match'),
    ('PG.N', 'PG', 'PG', 'Direct match'),
    ('PM.N', 'PM', 'PM', 'Direct match'),
    ('QCOM.OQ', 'QCOM', 'QCOM', 'Direct match'),
    ('RTX.N', 'RTX', 'UTX->RTX', 'United Technologies merged with Raytheon on 2020-04-03'),
    ('SBUX.OQ', 'SBUX', 'SBUX', 'Direct match'),
    ('SCHW.N', 'SCHW', 'SCHW', 'Direct match'),
    ('SO.N', 'SO', 'SO', 'Direct match'),
    ('SPG.N', 'SPG', 'SPG', 'Direct match'),
    ('T.N', 'T', 'T', 'Direct match'),
    ('TGT.N', 'TGT', 'TGT', 'Direct match'),
    ('TMO.N', 'TMO', 'TMO', 'Direct match'),
    ('TMUS.OQ', 'TMUS', 'TMUS', 'Direct match'),
    ('TSLA.OQ', 'TSLA', 'TSLA', 'Direct match'),
    ('TXN.OQ', 'TXN', 'TXN', 'Direct match'),
    ('UNH.N', 'UNH', 'UNH', 'Direct match'),
    ('UNP.N', 'UNP', 'UNP', 'Direct match'),
    ('UPS.N', 'UPS', 'UPS', 'Direct match'),
    ('USB.N', 'USB', 'USB', 'Direct match'),
    ('V.N', 'V', 'V', 'Direct match'),
    ('VZ.N', 'VZ', 'VZ', 'Direct match'),
    ('WFC.N', 'WFC', 'WFC', 'Direct match'),
    ('WMT.N', 'WMT', 'WMT', 'Direct match'),
    ('XOM.N', 'XOM', 'XOM', 'Direct match'),
]

# =============================================================================
# PRINT SUMMARY (when run directly)
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("CRSP TICKER MAPPING SUMMARY")
    print("="*80)
    
    print("\n STATIC MAPPINGS (name differences):")
    for k, v in STATIC_TICKER_MAP.items():
        print(f"   {k} -> {v}")
    
    print("\n TICKER CHANGES (date-based):")
    for ticker, changes in TICKER_CHANGES.items():
        print(f"   {ticker}:")
        for crsp, start, end in changes:
            end_str = end.strftime('%Y-%m-%d') if end else 'present'
            print(f"      {crsp}: {start.strftime('%Y-%m-%d')} to {end_str}")
    
    print("\n DATA GAPS (no CRSP data):")
    for ticker, gaps in DATA_GAPS.items():
        print(f"   {ticker}:")
        for start, end in gaps:
            print(f"      No data: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    print("\n✓ Mapping file ready for use")
