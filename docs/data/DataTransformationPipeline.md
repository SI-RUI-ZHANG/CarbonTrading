# Data Transformation Pipeline Documentation

## Overview

This document provides comprehensive documentation of the entire data transformation pipeline for the Chinese regional carbon markets research project. The pipeline processes raw carbon market data from Guangdong (GDEA) and Hubei (HBEA), combines it with 20 macroeconomic indicators, and produces feature-engineered datasets ready for modeling.

**Pipeline Structure:**
1. Carbon Markets Data Processing
2. Macroeconomic Indicators Processing  
3. Feature Engineering and Final Join

**Date Range**: 2012-01-01 to 2025-05-06  
**Final Output**: Two feature-rich datasets with 25 columns each for HBEA and GDEA markets

---

## Part 1: Carbon Markets Data Processing

### 1.1 GDEA Raw Data Processing
**Source**: `01_Data_Raw/01_Carbon_Markets/GDEA_raw.csv`  
**Process**: `03_Code/01_Data_Cleaning/01_Carbon_Markets/01_GDEA_Process.ipynb`  
**Output**: `02_Data_Processed/01_Carbon_Markets/01_Regional/GDEA_processed.parquet`

#### Input Data Structure
- **Source**: Wind Financial Terminal (广州碳排放权交易所)
- **Raw Shape**: 4,163 × 7 (including metadata rows)
- **Effective Observations**: 4,157 rows after removing 4 header and 2 footer rows
- **Date Range**: 2013-12-19 to 2025-05-06

#### Transformation Steps

1. **Metadata Removal & Type Casting**
   - Remove first 4 rows (frequency, unit, indicator ID, source)
   - Remove last 2 rows (data source attribution)
   - Rename columns to English:
     - `date`, `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`
   - Parse date column to datetime, numeric columns to float64

2. **Market Maturity Detection**
   - Calculate 60-day rolling trade frequency (percentage of days with trades)
   - Apply 20-day smoothing for visualization
   - **Mature start date identified**: 2014-06-27 (first date where 60-day frequency ≥ 50%)
   - Filter data to keep only mature phase (3,967 rows retained)

3. **Trading Calendar Alignment**
   - Use Shanghai Stock Exchange (XSHG) calendar via `exchange_calendars`
   - **Trading days**: 2,638 (66.5% of calendar days)
   - **Non-trading days**: 1,329 (33.5% of calendar days)
   - **Data quality findings**:
     - 15 non-trading days with activity (1.13% misalignment)
     - 232 trading days with no activity (8.79% quiet days)

4. **Flag Generation & Missing Value Treatment**
   - Generate three boolean flags:
     - `is_open`: True if date is XSHG trading day
     - `is_quiet`: True if `is_open` AND no trading activity
     - `has_trade`: True if `volume_tons` is not null
   - Fill missing values:
     - `volume_tons`, `turnover_cny`: Fill with 0.0
     - `cum_turnover_cny`: Forward fill
     - `close`, `vwap`: Keep as NaN (to be handled in next step)

#### Output Schema
| Column | Type | Description | Units |
|--------|------|-------------|-------|
| date (index) | datetime64 | Trading date | YYYY-MM-DD |
| close | float64 | Daily closing price (NaN if no trade) | CNY/ton |
| vwap | float64 | Volume-weighted average price | CNY/ton |
| volume_tons | float64 | Daily trading volume (0 if no trade) | tons |
| turnover_cny | float64 | Daily turnover (0 if no trade) | CNY |
| cum_turnover_cny | float64 | Cumulative turnover (forward filled) | CNY |
| is_open | bool | Shanghai Exchange trading day flag | - |
| is_quiet | bool | Open but no trades flag | - |
| has_trade | bool | Trading activity flag | - |

### 1.2 HBEA Raw Data Processing
**Source**: `01_Data_Raw/01_Carbon_Markets/HBEA_raw.csv`  
**Process**: `03_Code/01_Data_Cleaning/01_Carbon_Markets/02_HBEA_Process.ipynb`  
**Output**: `02_Data_Processed/01_Carbon_Markets/01_Regional/HBEA_processed.parquet`

#### Input Data Structure
- **Source**: Wind Financial Terminal (湖北碳排放权交易中心)
- **Raw Shape**: 4,033 × 7 (including metadata rows)
- **Effective Observations**: 4,027 rows after removing 4 header and 2 footer rows
- **Date Range**: 2014-04-28 to 2025-05-06

#### Transformation Steps

1. **Metadata Removal & Type Casting**
   - Remove first 4 rows (frequency, unit, indicator ID, source)
   - Remove last 2 rows (data source attribution)
   - Drop duplicate `cum_turnover_cny` column (column index 5)
   - Rename columns to English (same schema as GDEA)
   - Parse date column to datetime, numeric columns to float64

2. **Market Maturity Detection**
   - Calculate 60-day rolling trade frequency
   - **Key finding**: HBEA market mature from start (2014-06-26)
   - No filtering needed as mature_start equals earliest date + 59 days
   - All 4,027 rows retained

3. **Trading Calendar Alignment**
   - Use Shanghai Stock Exchange (XSHG) calendar
   - **Trading days**: 2,679 (66.5% of calendar days)
   - **Non-trading days**: 1,348 (33.5% of calendar days)
   - **Data quality findings**:
     - 0 non-trading days with activity (0% misalignment - better than GDEA)
     - 69 trading days with no activity (2.58% quiet days - better than GDEA)

4. **Flag Generation & Missing Value Treatment**
   - Generate same three boolean flags as GDEA
   - Fill missing values using identical strategy:
     - `volume_tons`, `turnover_cny`: Fill with 0.0
     - `cum_turnover_cny`: Forward fill
     - `close`, `vwap`: Keep as NaN

#### Output Schema
Same structure as GDEA with 4,027 rows × 8 columns plus index

#### Key Differences from GDEA
- **Earlier launch**: HBEA started April 2014 vs GDEA December 2013
- **Immediate maturity**: No pre-mature phase filtering needed
- **Better data quality**: 0% non-trading day misalignment, only 2.58% quiet trading days
- **Fewer observations**: 4,027 rows vs GDEA's 3,967 (due to later mature start for GDEA)

### 1.3 Price Imputation Strategies
**Source**: GDEA/HBEA processed files from steps 1.1 and 1.2  
**Process**: `03_Code/01_Data_Cleaning/01_Carbon_Markets/03_GDEAHBEA_forwardFill_interpolation.ipynb`  
**Outputs**: Four variants for each market (8 files total)

#### Imputation Methods

1. **Forward Fill Strategy**
   - **Method**: `pandas.DataFrame.ffill()` on price columns
   - **Rationale**: Assumes last traded price persists until next trade
   - **Application**: Both `close` and `vwap` columns
   - **Output files**:
     - `GDEA_forward_filled.parquet`
     - `HBEA_forward_filled.parquet`

2. **Linear Interpolation Strategy**
   - **Method**: `pandas.DataFrame.interpolate(method='linear', limit_direction='both')`
   - **Rationale**: Assumes smooth price transition between trades
   - **Rounding**: Results rounded to 1 decimal place
   - **Application**: Both `close` and `vwap` columns
   - **Output files**:
     - `GDEA_interpolated.parquet`
     - `HBEA_interpolated.parquet`

#### Output Location
All files saved to: `02_Data_Processed/01_Carbon_Markets/01_Regional/`

#### Missing Value Summary
| Market | Method | Close NaN Before | Close NaN After | VWAP NaN Before | VWAP NaN After |
|--------|--------|------------------|-----------------|-----------------|----------------|
| GDEA | Forward Fill | 1,548 (39.0%) | 0 | 1,546 (39.0%) | 0 |
| GDEA | Interpolation | 1,548 (39.0%) | 0 | 1,546 (39.0%) | 0 |
| HBEA | Forward Fill | 1,418 (35.2%) | 0 | 1,417 (35.2%) | 0 |
| HBEA | Interpolation | 1,418 (35.2%) | 0 | 1,417 (35.2%) | 0 |

#### Business Logic
- **Forward fill** preferred for trading strategies to avoid look-ahead bias
- **Interpolation** provides smoother price series for certain econometric models
- Both methods preserve the original flag columns for filtering if needed

---

## Part 2: Macroeconomic Indicators Processing

### 2.1 Raw Macroeconomic Data Cleaning
**Source**: `01_Data_Raw/02_Macroeconomic_Indicators/` (20 CSV files)  
**Process**: `03_Code/01_Data_Cleaning/02_Macroeconomic_Indicators/01_MacroData_Processes.ipynb`  
**Outputs**: Forward filled and interpolated variants in separate directories

#### Input Files (20 indicators)
```
广东_工业增加值_可比价_规模以上工业企业_当月同比.csv
湖北_工业增加值_可比价_规模以上工业企业_当月同比.csv
中国_产量_原油加工量_当月值.csv
中国_产量_原煤_当月值.csv
中国_产量_水泥_当月值.csv
中国_产量_粗钢_当月值.csv
中国_发电量_火电_当月值.csv
中国_全社会用电量_当月值.csv
中国_社会融资规模_当月值.csv
广东_用电量_当月值.csv
湖北_用电量_当月值.csv
中国_CPI_当月同比.csv
中国_GDP_现价_累计值.csv
广东_GDP_累计值.csv
湖北_GDP_累计值.csv
中国_制造业PMI.csv
期货结算价(连续)_布伦特原油.csv
期货结算价(连续)_欧盟排放配额(EUA).csv
期货收盘价(连续)_NYMEX天然气.csv
CFETS_即期汇率_美元兑人民币.csv
```

#### Transformation Steps

1. **Metadata Removal & Type Casting**
   - Skip first 8 rows (Wind metadata) and last 2 rows (footer)
   - Add column headers: `date`, `value`
   - Parse date to datetime, value to numeric
   - Filter date range: 2012-01-01 to 2025-04-29

2. **Frequency Detection**
   - Calculate median gap between consecutive dates
   - Classification logic:
     - **Daily (D)**: median gap = 1 day (5 indicators)
     - **Monthly (M)**: median gap ∈ [27, 32] days (13 indicators)
     - **Quarterly (Q)**: median gap ∈ [85, 95] days (3 indicators)

3. **Missing Value Imputation - Two Strategies**

   **Forward Fill Version** (`02_Data_Processed/02_Macroeconomic_Indicators/01_Forward_Filled/`)
   - Apply `pandas.DataFrame.ffill()` to value column
   - Add `value_filled` boolean flag for imputed values
   - File naming: `{indicator_name}_ffill.parquet`

   **Interpolation Version** (`02_Data_Processed/02_Macroeconomic_Indicators/02_Interpolated/`)
   - Apply `pandas.DataFrame.interpolate(method='linear', limit_direction='forward')`
   - Round to 1 decimal place
   - Add `value_filled` boolean flag
   - File naming: `{indicator_name}_interp.parquet`

#### Frequency Groups & Missing Value Statistics

| Frequency | Count | Indicators | Avg Missing % |
|-----------|-------|------------|---------------|
| Daily | 4 | FX rate, energy futures | 15.5% |
| Monthly | 13 | Industrial production, electricity, CPI, PMI | 4.5% |
| Quarterly | 3 | GDP (China, Guangdong, Hubei) | 0.0% |

#### Output Schema (All Files)
| Column | Type | Description |
|--------|------|-------------|
| date | datetime64 | Observation date |
| value | float64 | Indicator value (filled/interpolated) |
| value_filled | bool | True if value was imputed |

### 2.2 Daily Frequency Alignment
**Source**: Forward filled and interpolated files from step 2.1  
**Process**: `03_Code/01_Data_Cleaning/02_Macroeconomic_Indicators/02_MacroData_AlignDaily.ipynb`  
**Outputs**: Daily-aligned versions in two directories

#### Transformation Logic

1. **Forward Fill Daily Alignment** (`03_Forward_Filled_Daily/`)
   - Load each `_ffill.parquet` file
   - Create continuous daily date range from min to max date
   - Reindex to daily frequency
   - Apply forward fill to propagate values
   - Add `is_filled_daily` flag for newly created dates
   - File naming: `{indicator_name}_ffill_daily.parquet`

2. **Interpolation Daily Alignment** (`04_Interpolated_Daily/`)
   - Load each `_interp.parquet` file
   - Create continuous daily date range
   - Reindex to daily frequency
   - Apply linear interpolation for `value` column
   - Forward fill the `value_filled` flag (preserves original imputation info)
   - Add `is_interpolated_daily` flag for newly created dates
   - File naming: `{indicator_name}_interp_daily.parquet`

#### Key Differences Between Methods
| Aspect | Forward Fill | Interpolation |
|--------|--------------|---------------|
| Value propagation | Step function (constant between observations) | Linear transition |
| Flag handling | Forward fills all columns | Interpolates value, forward fills flags |
| Use case | Discrete changes (e.g., policy rates) | Continuous variables (e.g., prices) |

#### Output Schema (Daily-Aligned Files)
| Column | Type | Description |
|--------|------|-------------|
| date | datetime64 | Daily date index (no gaps) |
| value | float64 | Indicator value at daily frequency |
| value_filled | bool | Original imputation flag from step 2.1 |
| is_filled_daily / is_interpolated_daily | bool | True for dates added during daily alignment |

#### Row Counts
- **All files**: 4,868 rows (2012-01-01 to 2025-04-29)
- Original observations vary by frequency:
  - Daily indicators: ~4,000 original dates
  - Monthly indicators: ~160 original dates expanded to 4,868
  - Quarterly indicators: ~54 original dates expanded to 4,868

---

## Part 3: Feature Engineering

### 3.1 File Organization and Translation
**Source**: `02_Data_Processed/02_Macroeconomic_Indicators/03_Forward_Filled_Daily/`  
**Process**: `03_Code/02_Feature_Engineering/00_NameParse.py`  
**Output**: Reorganized files by geographic scope

#### Translation Dictionary (CN → EN)
Key translations include:
- 中国 → China
- 湖北 → Hubei  
- 广东 → Guangdong
- 用电量 → ElectricityConsumption
- 工业增加值 → IndustrialAddedValue
- GDP → GDP
- 产量 → Output
- 粗钢 → CrudeSteel
- 原油加工量 → CrudeOilProcessing
- 火电 → ThermalPower
- 制造业PMI → ManufacturingPMI

#### Geographic Classification Logic
Files are organized into three folders based on first component:
1. **`hubei/`**: Province-specific indicators (3 files)
   - Hubei_ElectricityConsumption_Monthly_ffill_daily
   - Hubei_IndustrialAddedValue_RealPrices_AboveScaleIndustry_YoY_ffill_daily
   - Hubei_GDP_Cumulative_ffill_daily

2. **`guangdong/`**: Province-specific indicators (3 files)
   - Guangdong_ElectricityConsumption_Monthly_ffill_daily
   - Guangdong_IndustrialAddedValue_RealPrices_AboveScaleIndustry_YoY_ffill_daily
   - Guangdong_GDP_Cumulative_ffill_daily

3. **`national_or_global/`**: National China and global indicators (14 files)
   - China indicators: CPI, PMI, GDP, industrial outputs, electricity
   - Global indicators: Brent crude, EU carbon futures, natural gas, FX rate

#### Processing Steps
1. Parse Chinese filename components using underscore delimiter
2. Translate each component via dictionary lookup
3. Detect geographic region from first component
4. Create target directory structure
5. Copy files with English names to appropriate folders

#### Output Structure
```
02_Data_Processed/02_Macroeconomic_Indicators/03_Forward_Filled_Daily/
├── hubei/
│   ├── Hubei_ElectricityConsumption_Monthly_ffill_daily.parquet
│   └── ... (3 files)
├── guangdong/
│   ├── Guangdong_ElectricityConsumption_Monthly_ffill_daily.parquet
│   └── ... (3 files)
└── national_or_global/
    ├── China_CPI_YoY_ffill_daily.parquet
    ├── FuturesSettle(Cont)_BrentCrude_ffill_daily.parquet
    └── ... (14 files)
```

### 3.2 Final Join with Intelligent Lag Structure
**Source**: Forward-filled carbon prices and organized macro indicators  
**Process**: `03_Code/02_Feature_Engineering/01_join_macro_15day_lag.ipynb`  
**Output**: `02_Data_Processed/03_Feature_Engineered/`

#### Input Files
- **Carbon Markets**: 
  - `HBEA_forward_filled.parquet`
  - `GDEA_forward_filled.parquet`
- **Macroeconomic Indicators**: 
  - 20 files from geographic folders (hubei/, guangdong/, national_or_global/)

#### Lag Strategy Based on Frequency Detection

The notebook implements intelligent lagging to prevent look-ahead bias:

1. **Frequency Detection Algorithm**
   ```python
   def detect_frequency(df):
       # Find where value changes (not forward-filled)
       changes = df['value'].ne(df['value'].shift())
       change_dates = df.index[changes]
       gaps = (change_dates[1:] - change_dates[:-1]).days
       median_gap = np.median(gaps)
   ```

2. **Lag Assignment**
   | Original Frequency | Median Gap | Applied Lag | Rationale |
   |-------------------|------------|-------------|-----------|
   | Daily (D) | ≤1.5 days | 1 day | Next-day availability |
   | Monthly (M) | 25-35 days | 15 days | Mid-month reporting delay |
   | Quarterly (Q) | 80-100 days | 15 days | Conservative reporting delay |

#### Join Process

1. **Load and Prepare Carbon Data**
   - Set date as index
   - Preserve all original columns (close, vwap, volume_tons, etc.)

2. **Process Macro Indicators**
   - Detect frequency for each indicator
   - Apply appropriate lag using `df.shift(periods=n)`
   - Rename columns: `{indicator}_{lag_days}` (e.g., `China_CPI_YoY_ffill_daily_15`)

3. **Geographic Joining**
   - **HBEA**: Join with ['hubei', 'national_global'] indicators
   - **GDEA**: Join with ['guangdong', 'national_global'] indicators
   - Method: Left join on date index to preserve all carbon market dates

#### Final Output Schema

**Files Created**:
- `HBEA_daily_with_macro.parquet` (4,027 rows × 25 columns)
- `GDEA_daily_with_macro.parquet` (3,967 rows × 25 columns)

**Column Structure**:
| Column Category | Count | Examples |
|-----------------|-------|----------|
| Carbon market | 8 | close, vwap, volume_tons, turnover_cny, cum_turnover_cny, is_open, is_quiet, has_trade |
| Province macro | 3 | {Province}_ElectricityConsumption_Monthly_ffill_daily_15, {Province}_IndustrialAddedValue_*, {Province}_GDP_* |
| National macro | 10 | China_CPI_YoY_ffill_daily_15, China_ManufacturingPMI_ffill_daily_15, China_Output_* |
| Global macro | 4 | FuturesSettle(Cont)_BrentCrude_ffill_daily_1, CFETS_SpotFX_USD_CNY_ffill_daily_1, etc. |

#### Data Quality Metrics
- **Missing values**: 503 total in each dataset (primarily from early dates before macro data starts)
- **Date coverage**: 
  - HBEA: 2014-04-28 to 2025-05-06
  - GDEA: 2014-06-27 to 2025-05-06
- **Trading days preserved**: All original carbon market dates retained

#### Business Logic Summary
- **No look-ahead bias**: All indicators lagged appropriately for realistic backtesting
- **Geographic relevance**: Province-specific indicators matched to respective markets
- **252 trading days assumption**: For annualized performance metrics
- **Data lineage preserved**: All processing flags maintained through pipeline

---

## Assumptions and Notes

1. **Trading Calendar**: Shanghai Stock Exchange (XSHG) calendar used as proxy for carbon market trading days
2. **Price Imputation**: Forward fill preferred over interpolation to avoid look-ahead bias in trading strategies
3. **Lag Structure**: 15-day lag for monthly/quarterly data assumes mid-month reporting typical in Chinese markets
4. **Missing Data**: Early period missing values (before 2012) accepted as macro data limitation
5. **File Format**: Parquet used throughout for efficient storage and type preservation
6. **Timezone**: All dates in local China time (UTC+8), no timezone information stored

## Pipeline Summary

**Total Files Processed**: 29 input files → 8 intermediate stages → 2 final outputs  
**Data Points**: ~100,000 daily observations across 26 features  
**Processing Time**: Full pipeline ~2-3 minutes on standard hardware  
**Storage**: ~50MB total for all processed files
