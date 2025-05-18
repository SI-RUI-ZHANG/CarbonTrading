# Data Card: Regional Carbon Markets (Processed)

## Overview

- **Description**: Cleaned daily trading records for Guangdong (GDEA) and Hubei (HBEA) emission allowances. Includes forward-filled and interpolated price variants.
- **Source**: Raw CSV files in `01_Data_Raw/01_Carbon_Markets` processed via notebooks `01_GDEA_Process.ipynb`, `02_HBEA_Process.ipynb`, and `03_GDEAHBEA_forwardFill_interpolation.ipynb`.
- **Date Range**:
  - GDEA: 2014-06-27 → 2025-05-06
  - HBEA: 2014-04-28 → 2025-05-06
- **Frequency**: Daily trading calendar (non-trading days included).
- **Files**: Each market saved as CSV and Parquet with three versions:
  - `<market>_processed.(csv|parquet)` – raw prices with trade flags and zero-filled activity columns.
  - `<market>_forward_filled.(csv|parquet)` – price columns forward filled.
  - `<market>_interpolated.(csv|parquet)` – price columns linearly interpolated.

## Dataset Description

### Columns

| Column             | Description                                   | Unit       |
| ------------------ | --------------------------------------------- | ---------- |
| `date`             | Trading or calendar date                      | YYYY‑MM‑DD |
| `close`            | Closing price of the allowance                | CNY/ton    |
| `vwap`             | Volume-weighted average price                 | CNY/ton    |
| `volume_tons`      | Daily traded volume                           | ton        |
| `turnover_cny`     | Daily traded value                            | CNY        |
| `cum_turnover_cny` | Cumulative turnover                           | CNY        |
| `is_open`          | True if the date is a trading day             | bool       |
| `is_quiet`         | True if trading day but `volume_tons` is zero | bool       |
| `has_trade`        | True if `volume_tons` > 0                     | bool       |
### Record Counts

- GDEA_processed: 3 967 rows
- HBEA_processed: 4 027 rows

Forward-filled and interpolated files share the same row counts.

## Data Processing Steps

1. **Clean Raw Structure**: Removed metadata/header rows and trailing footers, dropped duplicate columns, and standardized column names.
2. **Type Parsing**: Converted `date` to datetime and numeric columns to `float64`.
3. **Trading Calendar Alignment**: Reindexed to include all calendar days and created flags `is_open`, `is_quiet`, and `has_trade`.
4. **Activity Columns**: Filled missing `volume_tons` and `turnover_cny` with zeros; `cum_turnover_cny` forward filled.
5. **Price Handling**:
   - `*_processed`: Price columns left as-is (may contain gaps).
   - `*_forward_filled`: Filled price gaps forward.
   - `*_interpolated`: Linearly interpolated price gaps and rounded to one decimal.

## Assumptions

- Non-trading days carry zero volume and turnover.
- Cumulative turnover is assumed to increase monotonically and is forward filled on non-trading days.
- Forward fill and interpolation for prices do not extrapolate beyond the dataset bounds.
