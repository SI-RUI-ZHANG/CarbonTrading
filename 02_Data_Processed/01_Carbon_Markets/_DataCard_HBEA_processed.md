# Data Card: HBEA_processed.csv

## Overview

- **Dataset Name**: HBEA_processed
- **Description**: Daily trading data for the Hubei Carbon Emissions Allowance (HBEA) from the Hubei Carbon Emissions Trading Center, processed from raw data sourced from Wind. The dataset covers the full market period (2014-04-28 to 2025-05-06) and includes price, volume, turnover, and cumulative turnover metrics, with imputed values for missing data based on specific rules.
- **Source**: Raw data from Wind Financial Terminal, processed using Python.
- **Date Range**: 2014-04-28 to 2025-05-06
- **File Format**: CSV (`HBEA_processed.csv`) and Parquet (`HBEA_processed.parquet`)

## Dataset Description

### Columns

- **`date`**:
  - **Description**: Date of the record (index)
  - **Data Type**: Datetime
  - **Units**: YYYY-MM-DD
  - **Notes**: Set as index, sorted ascending.
- **`close`**:
  - **Description**: Closing price of HBEA
  - **Data Type**: Float64
  - **Units**: CNY/ton
  - **Notes**: Forward-filled for one trading day on open days; 46 NaNs remain.
- **`vwap`**:
  - **Description**: Volume-weighted average price of HBEA
  - **Data Type**: Float64
  - **Units**: CNY/ton
  - **Notes**: Forward-filled for one trading day on open days; 46 NaNs remain.
- **`volume_tons`**:
  - **Description**: Trading volume in tons of carbon allowances
  - **Data Type**: Float64
  - **Units**: Tons
  - **Notes**: Set to 0.0 for open days with no trades; NaN on non-trading days.
- **`turnover_cny`**:
  - **Description**: Daily turnover in Chinese Yuan
  - **Data Type**: Float64
  - **Units**: CNY
  - **Notes**: Set to 0.0 for open days with no trades; NaN on non-trading days.
- **`cum_turnover_cny`**:
  - **Description**: Cumulative turnover in Chinese Yuan
  - **Data Type**: Float64
  - **Units**: CNY
  - **Notes**: Forward-filled without limit; no NaNs remain on open days.
- **`is_open`**:
  - **Description**: Flag indicating if the day is a trading day
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: Based on XSHG calendar; True for 2,679 days, False for 1,348 days.
- **`is_quiet`**:
  - **Description**: Flag indicating open days with no trading activity
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for 69 trading days with no trades (NaN in raw `volume_tons`).
- **`has_trade`**:
  - **Description**: Flag indicating days with trading activity
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for 2,610 days with non-NaN `volume_tons` in raw data.

### Row Counts

- **Total Rows**: 4,027 (daily records from 2014-04-28 to 2025-05-06)
- **Trading Days (`is_open == True`)**: 2,679 (66.5%)
- **Non-Trading Days (`is_open == False`)**: 1,348 (33.5%)
- **Days with Trading Activity (`has_trade == True`)**: 2,610 (64.8%)
- **Quiet Trading Days (`is_quiet == True`)**: 69 (2.6% of trading days)
- **Non-Trading Days with Activity**: 0 (0.0% of non-trading days)

### Missing Values (Post-Processing)

- `close`: 46 NaNs (1.14% of rows, all on trading days)
- `vwap`: 46 NaNs (1.14% of rows, all on trading days)
- `volume_tons`: 1,348 NaNs (33.5%, all on non-trading days)
- `turnover_cny`: 1,348 NaNs (33.5%, all on non-trading days)
- `cum_turnover_cny`: 0 NaNs (0%, fully filled)
- `is_open`, `is_quiet`, `has_trade`: 0 NaNs (0%, fully populated)

## Data Processing Steps

The dataset was processed from the raw file `HBEA_raw.csv` using a Python script (`02_HBEA_Process.ipynb`). The following steps were applied:

1. **Data Loading**:
   - Loaded raw CSV (`../../01_Data_Raw/01_Carbon_Markets/HBEA_raw.csv`) using `pandas` with `gb18030` encoding.
   - Raw shape: 4,033 rows, 7 columns (including metadata and duplicate column).

2. **Data Cleaning**:
   - Removed metadata (first 4 rows) and footer (last 2 rows).
   - Dropped duplicate column (`湖北:累计成交量:碳排放权(HBEA).1`).
   - Renamed columns to: `date`, `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`.
   - Sorted by `date` (ascending).
   - Converted `date` to `datetime` and set as index.
   - Converted numeric columns to `float64`, with non-numeric values coerced to NaN.

3. **Mature Phase Filtering**:
   - Calculated 60-day rolling trade frequency (fraction of days with non-NaN `volume_tons`).
   - Identified mature start date (2014-06-26) as the first date with ≥50% trade frequency.
   - Determined the market was mature from the start (2014-04-28), so no rows were filtered (retained 4,027 rows).

4. **Trading Day Identification**:
   - Used the Shanghai Stock Exchange (XSHG) calendar from `exchange_calendars` to label trading days (`is_open`).
   - Identified 2,679 trading days and 1,348 non-trading days.
   - Added flags:
     - `is_quiet`: True for trading days with NaN `volume_tons` (69 days).
     - `has_trade`: True for days with non-NaN `volume_tons` (2,610 days).

5. **Missing Value Imputation**:
   - **Activity Columns (`volume_tons`, `turnover_cny`)**:
     - Set to 0.0 for trading days with NaN values (`is_quiet == True`).
     - Left as NaN for non-trading days.
   - **Price Columns (`close`, `vwap`)**:
     - Forward-filled for one trading day within `is_open` blocks.
     - 46 NaNs remain for both `close` and `vwap` (unfilled due to consecutive quiet days or early data gaps).
   - **Cumulative Turnover (`cum_turnover_cny`)**:
     - Forward-filled without limit to ensure continuity.
     - No NaNs remain on trading days.
   - **Non-Trading Days with Activity**:
     - Identified 0 non-trading days with non-NaN `volume_tons` (0.0% of non-trading days).

6. **Clean-Up**:
   - Dropped redundant `is_trading_day` column (identical to `is_open`).
   - Ensured all numeric columns are `float64`.
   - No rows were dropped, preserving the full date range (no rows with NaN prices were explicitly dropped, despite the stated strategy).

7. **Persistence**:
   - Saved as `HBEA_processed.csv` and `HBEA_processed.parquet` in `../../02_Data_Processed/01_Carbon_Markets/`.
   - Visualized inspection results (pie charts for trading and non-trading day statuses).

## Assumptions

- **Calendar Accuracy**: The XSHG calendar is assumed to approximate the Hubei Carbon Emissions Trading Center's trading schedule. No non-trading days had activity, suggesting better calendar alignment than for other markets (e.g., GDEA).
- **No Trading Activity**: Trading days with NaN `volume_tons` in the raw data (69 days) are interpreted as days with no trades (set to 0.0 for `volume_tons` and `turnover_cny`).
- **Price Continuity**: Forward-filling `close` and `vwap` for one trading day assumes prices remain stable for a single day of no trading. Longer gaps result in NaNs.
- **Cumulative Turnover**: Forward-filling `cum_turnover_cny` assumes the cumulative total remains constant until new trading activity occurs.
- **Data Integrity**: The raw data from Wind is assumed to be mostly accurate, with NaNs reflecting either no trading or missing data rather than errors.

## Limitations

- **Calendar Uncertainty**: The XSHG calendar may not perfectly align with the Hubei Carbon Emissions Trading Center's schedule, though no non-trading day anomalies were detected. No official HBEA calendar was available during processing.
- **Remaining NaNs**: 46 NaNs in both `close` and `vwap` persist on trading days, indicating gaps not filled by forward-filling (e.g., consecutive quiet days).
- **No Row Dropping**: The strategy to drop rows with NaN prices was not implemented, retaining all 4,027 rows, which may include sparse early data.
- **Early Data Sparsity**: The period around 2014-04-28 may have lower data quality due to the market's initial phase, despite the immediate ≥50% trade frequency.
- **No External Validation**: Imputation rules were not cross-checked with Hubei Carbon Emissions Trading Center reports or Wind metadata due to lack of access.