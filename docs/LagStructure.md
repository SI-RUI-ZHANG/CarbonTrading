# Lag Structure

The system implements two distinct lag layers: data availability lags in the pipeline and execution lag in strategies.

## Two-Layer Architecture

### 1. Data Pipeline Lags (`01_join_macro_15day_lag.ipynb`)
Applied to macroeconomic indicators based on reporting frequency:
- **Daily indicators** (FX, futures): 1-day lag
- **Monthly/Quarterly indicators** (CPI, GDP, PMI): 15-day lag
- **Carbon prices**: No lag (available at close)

### 2. Strategy Execution Lag (`a_Strategy.py`)
Applied to ALL signals via `signal.shift(fill_value=0)`:
- **All data sources**: 1-day lag for trade execution

## Total Lag by Data Type

| Data Type                 | Pipeline Lag | Execution Lag | Total   |
| ------------------------- | ------------ | ------------- | ------- |
| Carbon prices             | 0 days       | 1 day         | 1 day   |
| Daily macro (FX, futures) | 1 day        | 1 day         | 2 days  |
| Monthly macro (CPI, PMI)  | 15 days      | 1 day         | 16 days |
| Quarterly macro (GDP)     | 15 days      | 1 day         | 16 days |

## Timeline Example

Trading on May 20th with CPI-based signal:
```
May 5:  CPI value recorded
May 20: CPI available in dataset (15-day lag)
        Strategy generates signal using May 5 CPI + today's carbon price
May 21: Trade executes (1-day execution lag)
```

## Key Point

Carbon market data has no pipeline lag because closing prices are immediately observable. Macro data has pipeline lags reflecting real-world reporting delays. Both get the execution lag because signals calculated after close can only be traded the next day.