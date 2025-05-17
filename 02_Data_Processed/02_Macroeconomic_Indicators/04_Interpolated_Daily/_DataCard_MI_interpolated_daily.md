# Data Card: Macroeconomic Indicators (Interpolated Daily)

## Overview

- **Dataset Name**: MacroIndicators_Interpolated_Daily
- **Description**: Daily-aligned version of the interpolated macroeconomic series. Missing days are linearly interpolated and flagged with `is_interpolated_daily`.
- **Source**: Parquet files in `02_Data_Processed/02_Macroeconomic_Indicators/02_Interpolated` processed with `02_MacroData_AlignDaily.ipynb`.
- **Date Range**: 2012-01-01 to 2025-04-29
- **File Format**: Parquet only; each series stored in `04_Interpolated_Daily`.

## Dataset Description

### Columns

- **`date`**:
  - **Description**: Daily date index.
  - **Data Type**: Datetime
  - **Units**: YYYY-MM-DD
  - **Notes**: Full daily range with no gaps.
- **`value`**:
  - **Description**: Indicator value after interpolation to daily frequency.
  - **Data Type**: float64
  - **Units**: Varies by indicator
  - **Notes**: Linear interpolation across missing days.
- **`value_filled`**:
  - **Description**: Flag from the source file marking interpolated values.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: Carried through from `02_Interpolated`.
- **`is_interpolated_daily`**:
  - **Description**: Flag showing rows added during daily alignment and interpolation.
  - **Data Type**: Boolean
  - **Units**: True/False
  - **Notes**: True for dates not present in the source index.

### Row Counts

- **All Series**: 4,868 rows each covering every day in the date range.

### Files

Parquet filenames follow the pattern `<name>_interp_daily.parquet` (21 files total).

## Data Processing Steps

1. **Load Source Data**: Read each `<name>_interp.parquet` file from the interpolated directory.
2. **Date Expansion**: Generated a continuous daily index covering the entire date range.
3. **Interpolation**: Reindexed to the daily index and linearly interpolated missing values.
4. **Flagging**: Marked rows absent from the original index as `is_interpolated_daily`.
5. **Persistence**: Saved each dataset to the `04_Interpolated_Daily` folder as Parquet.

## Assumptions

- Linear interpolation reasonably approximates values between observations.
- Daily alignment assumes no data exists outside the provided range.