# Data Card: Macroeconomic Indicators (Interpolated)

## Overview

- **Dataset Name**: MacroIndicators_Interpolated
- **Description**: Collection of 21 macroeconomic time series cleaned and interpolated. Data covers daily, monthly and quarterly frequencies. Each series retains the original date index and includes a boolean column to indicate interpolated values.
- **Source**: Raw CSV files in `01_Data_Raw/02_Macroeconomic_Indicators` processed using the notebook `01_MacroData_Processes.ipynb`.
- **Date Range**: 2012-01-01 to 2025-04-29
- **File Format**: Each series saved as CSV and Parquet in `02_Data_Processed/02_Macroeconomic_Indicators/02_Interpolated`. The file `_fileNames.txt` lists all output files.

## Dataset Description

### Columns

- **`date`**:
  - **Description**: Observation date.
  - **Data Type**: Datetime
  - **Units**: variable frequency
  - **Notes**: Sorted ascending and preserved as provided.
- **`value`**:
  - **Description**: Original indicator value (may contain missing values before processing).
  - **Data Type**: float64
  - **Units**: Varies by indicator
  - **Notes**: Interpolated where gaps occur.
- **`value_filled`**:
  - **Description**: Flag indicating whether the corresponding `value` was interpolated.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: Added during processing using `pandas.DataFrame.interpolate`.

### Frequency Groups and Typical Row Counts

- **Daily Series**: 5 indicators (e.g., FX rate and energy/futures prices) with ~4,868 rows each.
- **Monthly Series**: 13 indicators such as industrial production and power generation with ~159 rows each.
- **Quarterly Series**: 3 GDP series with ~53 rows each.

### Files

The dataset includes 21 files. Example entries from `_fileNames.txt`:
```
广东_工业增加值_可比价_规模以上工业企业_当月同比_interp.parquet
湖北_工业增加值_可比价_规模以上工业企业_当月同比_interp.parquet
中国_产量_原油加工量_当月值_interp.parquet
... (total 21 names)
```

## Data Processing Steps

1. **Initial Cleanup**: Removed the first 8 metadata rows and last 2 footer rows from each raw CSV. Converted the first column to datetime and the second column to numeric, filtering the date range to 2012‑01‑01 through 2025‑04‑29.
2. **Interim Storage**: Saved cleaned tables to `_Interim1` Parquet files.
3. **Frequency Detection**: Inferred each series frequency (daily, monthly or quarterly) with a helper function (`infer_freq`).
4. **Interpolation**: Linearly interpolated missing values column by column, rounded results to one decimal place, and flagged them in `value_filled`.
5. **Persistence**: Saved each series as `<name>_interp.csv` and `<name>_interp.parquet` and recorded the filenames in `_fileNames.txt`.

## Assumptions

- Missing entries are assumed to follow a linear trend between surrounding observations.
- No scaling or seasonal adjustments were applied beyond interpolation.
- The provided date range is considered complete for the analysis period.

