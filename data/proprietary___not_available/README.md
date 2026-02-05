# Proprietary Data Directory

This directory contains proprietary data files that cannot be distributed with the repository due to licensing restrictions.

## Required Files

Place the following files in this directory:

| File | Description | Source |
|------|-------------|--------|
| `CRSP_SP100_ReturnData.db` | CRSP daily returns for S&P 100 constituents | CRSP via WRDS |
| `S&P100_price_hard.sqlite` | Daily adjusted close prices | Refinitiv Eikon |
| `S&P100_sectors_hard.sqlite` | GICS sector classifications | Refinitiv Eikon |
| `S&P100_volume_hard.sqlite` | Daily trading volume | Refinitiv Eikon |
| `S&P100_market_cap_hard.sqlite` | Market capitalization | Refinitiv Eikon |

## Database Schemas

### CRSP_SP100_ReturnData.db
- Table: `daily_returns`
- Columns: `TICKER`, `date`, `RET` (daily return as decimal)

### S&P100_*.sqlite files
- Table: `data`
- Columns: `instrument`, `date`, `value`

## Notes

- All dates should be in YYYY-MM-DD format
- Returns should be decimal (e.g., 0.01 = 1%)
- If you have data in a different format, update `scripts/config.py` accordingly
