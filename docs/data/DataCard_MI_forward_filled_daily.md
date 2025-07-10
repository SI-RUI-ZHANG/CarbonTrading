# Data Card: Macroeconomic Indicators (Forward Filled Daily)

## Overview

- **Dataset Name**: MacroIndicators_Forward_Filled_Daily
- **Description**: Daily frequency versions of 21 macroeconomic series. Values were forward filled from the original frequency to cover every day in the range. A boolean column marks the rows inserted during daily alignment.
- **Source**: Forward-filled files in `02_Data_Processed/02_Macroeconomic_Indicators/01_Forward_Filled` processed using the notebook `02_MacroData_AlignDaily.ipynb`.
- **Date Range**: 2012-01-01 to 2025-04-29
- **File Format**: Each series saved as Parquet only in `02_Data_Processed/02_Macroeconomic_Indicators/03_Forward_Filled_Daily`.

## Dataset Description

### Columns

- **`date`**:
  - **Description**: Observation date.
  - **Data Type**: Datetime (daily index)
  - **Units**: YYYY-MM-DD
  - **Notes**: Continuous daily range without gaps.
- **`value`**:
  - **Description**: Indicator value carried forward from the most recent observation.
  - **Data Type**: float64
  - **Units**: varies by indicator
  - **Notes**: Forward filled to match a daily calendar.
- **`value_filled`**:
  - **Description**: Flag indicating whether the original value was forward filled at its native frequency.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: Copied from the 01_Forward_Filled dataset.
- **`is_filled_daily`**:
  - **Description**: Flag indicating rows created during daily alignment.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for days not present in the source file.

### Row Counts

- Each file contains 4,868 rows covering the full date range.

### Files

The folder contains 21 Parquet files named `<indicator>_ffill_daily.parquet`.

## Data Processing Steps

1. **Load Forward-Filled Data**: Read each `<name>_ffill.parquet` file.
2. **Reindex to Daily**: Created a complete daily date range from the first to last observation and reindexed the DataFrame.
3. **Forward Fill**: Filled any resulting gaps using the last available value.
4. **Flag Inserted Rows**: Added `is_filled_daily` to mark dates not in the original data.
5. **Persistence**: Saved as `<name>_ffill_daily.parquet` in the output directory.

## Assumptions

- Indicator values remain constant between reported dates.
- The daily date range is inclusive of all days between the first and last observations.
- No interpolation beyond forward fill was performed after reindexing.