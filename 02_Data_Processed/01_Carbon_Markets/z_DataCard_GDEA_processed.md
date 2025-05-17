# Data Card: GDEA_processed.csv

## Overview

- **Dataset Name**: GDEA_processed
- **Description**: Daily trading data for the Guangdong Carbon Emissions Allowance (GDEA) from the Guangzhou Emissions Exchange, processed from raw data sourced from Wind. The dataset covers the mature phase of the market (2014-06-27 to 2025-05-06) and includes price, volume, turnover, and cumulative turnover metrics, with imputed values for missing data based on specific rules.
- **Source**: Raw data from Wind Financial Terminal, processed using Python.
- **Date Range**: 2014-06-27 to 2025-05-06
- **File Format**: CSV (`GDEA_processed.csv`) and Parquet (`GDEA_processed.parquet`)

## Dataset Description

### Columns

- **`date`**:
  - **Description**: Date of the record (index).
  - **Data Type**: Datetime
  - **Units**: YYYY-MM-DD
  - **Notes**: Set as index, sorted ascending.
- **`close`**:
  - **Description**: Closing price of GDEA.
  - **Data Type**: `{float64}`
  - **Units**: CNY/ton
  - **Notes**: Forward-filled for one trading day on open days; 125 NaNs remain.
- **`vwap`**:
  - **Description**: Volume-weighted average price of GDEA.
  - **Data Type**: `{float64}`
  - **Units**: CNY/ton
  - **Notes**: Forward-filled for one trading day on open days; 124 NaNs remain.
- **`volume_tons`**:
  - **Description**: Trading volume in tons of carbon allowances.
  - **Data Type**: `{float64}`
  - **Units**: Tons
  - **Notes**: Set to 0.0 for open days with no trades; NaN on non-trading days.
- **`turnover_cny`**:
  - **Description**: Daily turnover in Chinese Yuan.
  - **Data Type**: `{float64}`
  - **Units**: CNY
  - **Notes**: Set to 0.0 for open days with no trades; NaN on non-trading days.
- **`cum_turnover_cny`**:
  - **Description**: Cumulative turnover in Chinese Yuan.
  - **Data Type**: `{float64}`
  - **Units**: CNY
  - **Notes**: Forward-filled without limit; no NaNs remain on open days.
- **`is_open`**:
  - **Description**: Flag indicating if the day is a trading day.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: Based on XSHG calendar; True for 2,638 days, False for 1,329 days.
- **`is_quiet`**:
  - **Description**: Flag indicating open days with no trading activity.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for 232 trading days with no trades (NaN in raw `volume_tons`).
- **`has_trade`**:
  - **Description**: Flag indicating days with trading activity.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for 2,421 days with non-NaN `volume_tons` in raw data.

### Row Counts

- **Total Rows**: 3,967 (daily records from 2014-06-27 to 2025-05-06)
- **Trading Days (`is_open == True`)**: 2,638 (66.5%)
- **Non-Trading Days (`is_open == False`)**: 1,329 (33.5%)
- **Days with Trading Activity (`has_trade == True`)**: 2,421 (61.0%)
- **Quiet Trading Days (`is_quiet == True`)**: 232 (8.8% of trading days)
- **Non-Trading Days with Activity**: 15 (1.13% of non-trading days, potential calendar errors)

### Missing Values (Post-Processing)

- `close`: 125 NaNs (3.15% of rows, all on trading days)
- `vwap`: 124 NaNs (3.13% of rows, all on trading days)
- `volume_tons`: 1,329 NaNs (33.5%, all on non-trading days)
- `turnover_cny`: 1,329 NaNs (33.5%, all on non-trading days)
- `cum_turnover_cny`: 0 NaNs (0%, fully filled)
- `is_open`, `is_quiet`, `has_trade`: 0 NaNs (0%, fully populated)

## Data Processing Steps

The dataset was processed from the raw file `GDEA_raw.csv` using a Python script (`01_GDEA_Process.ipynb`). The following steps were applied:

1. **Data Loading**:
   - Loaded raw CSV (`../../01_Data_Raw/01_Carbon_Markets/GDEA_raw.csv`) using `pandas` with `gb18030` encoding.
   - Raw shape: 4,163 rows, 7 columns (including metadata and duplicate column).

2. **Data Cleaning**:
   - Removed metadata (first 4 rows) and footer (last 2 rows).
   - Dropped duplicate column (`广东:当日成交额:碳排放权配额(GDEA).1`).
   - Renamed columns to: `date`, `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`.
   - Sorted by `date` (ascending).
   - Converted `date` to `datetime` and set as index.
   - Converted numeric columns to `float64`, with non-numeric values coerced to NaN.

3. **Mature Phase Filtering**:
   - Calculated 60-day rolling trade frequency (fraction of days with non-NaN `volume_tons`).
   - Identified mature start date (2014-06-27) as the first date with ≥50% trade frequency.
   - Filtered data to include only 2014-06-27 to 2025-05-06 (3,967 rows).

4. **Trading Day Identification**:
   - Used the Shanghai Stock Exchange (XSHG) calendar from `exchange_calendars` to label trading days (`is_open`).
   - Identified 2,638 trading days and 1,329 non-trading days.
   - Added flags:
     - `is_quiet`: True for trading days with NaN `volume_tons` (232 days).
     - `has_trade`: True for days with non-NaN `volume_tons` (2,421 days).

5. **Missing Value Imputation**:
   - **Activity Columns (`volume_tons`, `turnover_cny`)**:
     - Set to 0.0 for trading days with NaN values (`is_quiet == True`).
     - Left as NaN for non-trading days.
   - **Price Columns (`close`, `vwap`)**:
     - Forward-filled for one trading day within `is_open` blocks.
     - 125 NaNs remain for `close`, 124 for `vwap` (unfilled due to consecutive quiet days or early data gaps).
   - **Cumulative Turnover (`cum_turnover_cny`)**:
     - Forward-filled without limit to ensure continuity.
     - No NaNs remain on trading days.
   - **Non-Trading Days with Activity**:
     - Identified 15 non-trading days with non-NaN `volume_tons` (1.13% of non-trading days).
     - Left unchanged (potential calendar errors, not imputed).

6. **Clean-Up**:
   - Dropped redundant `is_trading_day` column (identical to `is_open`).
   - Ensured all numeric columns are `float64`.
   - No rows were dropped, preserving the full date range.

7. **Persistence**:
   - Saved as `GDEA_processed.csv` and `GDEA_processed.parquet` in `../../02_Data_Processed/01_Carbon_Markets/`.
   - Visualized inspection results (pie charts for trading and non-trading day statuses).

## Assumptions

- **Calendar Accuracy**: The XSHG calendar is assumed to approximate the Guangzhou Emissions Exchange's trading schedule. This may lead to misclassification:
  - 15 non-trading days with activity (1.13%) may indicate calendar errors or data misalignments.
  - 232 trading days with all NaNs (8.8%) are assumed to be open market days with no trading activity.
- **No Trading Activity**: Trading days with NaN `volume_tons` in the raw data are interpreted as days with no trades (set to 0.0 for `volume_tons` and `turnover_cny`).
- **Price Continuity**: Forward-filling `close` and `vwap` for one trading day assumes prices remain stable for a single day of no trading. Longer gaps result in NaNs.
- **Cumulative Turnover**: Forward-filling `cum_turnover_cny` assumes the cumulative total remains constant until new trading activity occurs.
- **Data Integrity**: The raw data from Wind is assumed to be mostly accurate, with NaNs reflecting either no trading or missing data rather than errors (except for the 15 non-trading day anomalies).

## Limitations

- **Calendar Uncertainty**: The XSHG calendar may not perfectly align with the Guangzhou Emissions Exchange's schedule, potentially misclassifying trading/non-trading days. No official GDEA calendar was available during processing.
- **Remaining NaNs**: 125 NaNs in `close` and 124 in `vwap` persist on trading days, indicating gaps not filled by forward-filling (e.g., consecutive quiet days or early data sparsity).
- **Non-Trading Day Anomalies**: 15 non-trading days with activity were not corrected, pending confirmation of the correct calendar or data errors.
- **Early Data Sparsity**: The period shortly after 2014-06-27 may have lower data quality due to the market's immature phase, despite the ≥50% trade frequency threshold.
- **No External Validation**: Imputation rules were not cross-checked with Guangzhou Emissions Exchange reports or Wind metadata due to lack of access.