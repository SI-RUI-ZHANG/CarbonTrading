# LSTM Data Preparation

## Purpose

Add "data age" features to track the freshness of macroeconomic indicators. Since macro data is forward-filled between observations, LSTM models need to distinguish between actual data persistence and staleness from forward-filling.

## Implementation

### Data Age Calculation

For each macroeconomic column:
- Age = 0 when value changes (fresh observation)
- Age increments daily while value remains constant (forward-filled)

```python
def add_data_age(df, column):
    value_changed = df[column].ne(df[column].shift())
    age = (~value_changed).groupby(value_changed.cumsum()).cumsum()
    return age
```

### Column Selection

**Skip (always fresh):**
- Carbon market: `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`, `is_open`, `is_quiet`, `has_trade`

**Add age tracking:**
- All macro indicators (columns ending in `_1` or `_15`)

## Output Structure

### File Naming
- Input: `HBEA_daily_with_macro.parquet`
- Output: `HBEA_LSTM_ready.parquet`

### Column Convention
For each macro indicator, add corresponding age column:
- Original: `China_CPI_YoY_ffill_daily_15`
- Age column: `China_CPI_YoY_ffill_daily_15_age`

### Example Data

| date | China_CPI_YoY_ffill_daily_15 | China_CPI_YoY_ffill_daily_15_age |
|------|------------------------------|-----------------------------------|
| 2024-01-01 | 2.5 | 0 |
| 2024-01-02 | 2.5 | 1 |
| 2024-01-03 | 2.5 | 2 |
| ... | ... | ... |
| 2024-01-31 | 2.5 | 30 |
| 2024-02-01 | 2.8 | 0 |
| 2024-02-02 | 2.8 | 1 |

## Age Patterns by Frequency

| Indicator Type | Typical Max Age | Reset Frequency |
|----------------|-----------------|-----------------|
| Daily (FX, futures) | 3-5 days | Weekends/holidays |
| Monthly (CPI, PMI) | 28-31 days | Monthly |
| Quarterly (GDP) | 89-92 days | Quarterly |

## Why This Matters for LSTM

1. **Temporal awareness**: Model learns that 30-day-old CPI differs from fresh CPI
2. **Feature importance**: Can down-weight stale data automatically
3. **Pattern recognition**: Distinguishes market reaction to new data vs continued trends

## Usage in LSTM Models

```python
# Load LSTM-ready data
df = pd.read_parquet("HBEA_LSTM_ready.parquet")

# Features include both value and age
features = df.columns.tolist()
# ['close', 'China_CPI_YoY_ffill_daily_15', 'China_CPI_YoY_ffill_daily_15_age', ...]

# LSTM can learn patterns like:
# - Fresh CPI (age=0) → stronger price reaction
# - Stale CPI (age>20) → less predictive power
```

## Files

- **Notebook**: `03_Code/02_Feature_Engineering/02_add_data_age_LSTM.ipynb`
- **Output**: `02_Data_Processed/03_Feature_Engineered/[HBEA|GDEA]_LSTM_ready.parquet`

## Summary Statistics

- Original features: 25 (8 carbon + 17 macro)
- Added age features: 17 (macro only)
- Total LSTM features: 42