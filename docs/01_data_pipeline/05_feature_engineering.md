# LSTM Advanced Features Documentation

## Overview

This document details the advanced technical and temporal features added to enhance LSTM modeling capabilities for carbon price prediction.

**Purpose**: Capture market dynamics, temporal patterns, and technical indicators  
**Input**: LSTM-ready data with 48 features (including age tracking)  
**Output**: LSTM-advanced data with 50 features  
**Implementation**: `03_Code/04_LSTM/02_add_advanced_features_LSTM.ipynb`

## Feature Categories

### 1. Cyclical Time Encoding (4 features)

Temporal patterns encoded using sine/cosine transformations to preserve continuity.

#### Day of Week
- **dow_sin**: `np.sin(2 * np.pi * df.index.dayofweek / 7)`
- **dow_cos**: `np.cos(2 * np.pi * df.index.dayofweek / 7)`

**Rationale**: Captures weekly trading patterns while maintaining continuity (Sunday→Monday transition is smooth). Monday (0) maps to (sin=0, cos=1), Sunday (6) maps to (sin=-0.782, cos=0.623).

#### Month of Year
- **month_sin**: `np.sin(2 * np.pi * (df.index.month - 1) / 12)`
- **month_cos**: `np.cos(2 * np.pi * (df.index.month - 1) / 12)`

**Rationale**: Captures seasonal effects in carbon markets. January maps to (sin=0, cos=1), December maps to (sin=-0.5, cos=0.866), ensuring smooth year-end transition.

**Why cyclical > one-hot encoding**: Preserves distance relationships (December is close to January) and reduces dimensionality (2 features vs 7 for weekdays or 12 for months).

### 2. Gap Information (preserved from data cleaning)

The `gap_days` feature from the data cleaning step captures calendar days between consecutive trading days, effectively preserving weekend and holiday information in the trading-only dataset.

**Distribution**:
- 1: Regular weekday (most common)
- 3: Weekend gap (typical Monday)
- 4-11: Holiday periods
- Max observed: 11 days (extended holidays)

**Purpose**: Captures market closure patterns and helps predict potential price adjustments after non-trading periods without needing separate weekend/holiday flags.

### 3. Price Returns (2 features)

#### log_return
Daily logarithmic returns for better statistical properties.

```python
log_return = np.log(df['close'] / df['close'].shift(1))
```

- Mean: 0.000193 (slight positive drift)
- Std: 0.027 (2.7% daily volatility)
- Calculated between consecutive trading days only

#### return_5d
5-trading-day percentage returns for medium-term momentum.

```python
return_5d = df['close'].pct_change(5)
```

- Mean: 0.002 (0.2% weekly return)
- Captures momentum over exactly 5 trading days

### 4. Volatility Indicators (1 feature)

#### bb_width
Normalized Bollinger Band width as volatility proxy.

```python
sma20 = df['close'].rolling(20, min_periods=1).mean()
std20 = df['close'].rolling(20, min_periods=1).std()
upper_band = sma20 + 2 * std20
lower_band = sma20 - 2 * std20
bb_width = (upper_band - lower_band) / sma20
```

- Mean: 0.130 (13% average band width)
- Std: 0.117
- Rolling window uses exactly 20 trading days
- **Interpretation**: Higher values indicate increased volatility/uncertainty

### 5. Technical Momentum (1 feature)

#### rsi_14
14-trading-day Relative Strength Index using Wilder's smoothing.

```python
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta.clip(upper=0))
avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
rs = avg_gain / avg_loss
rsi_14 = 100 - (100 / (1 + rs))
```

- Range: [8.6, 92.5]
- Mean: 50.29 (neutral market)
- Calculated over 14 trading days
- **Thresholds**: >70 = overbought, <30 = oversold

### 6. Volume Analysis (1 feature)

#### volume_sma_20
20-trading-day simple moving average of trading volume.

```python
volume_sma_20 = df['volume_tons'].rolling(20, min_periods=1).mean()
```

- Mean: 37,237 tons
- Calculated over exactly 20 trading days
- **Purpose**: Baseline for detecting unusual trading activity

## Feature Statistics Summary

| Feature | Mean | Std | Min | Max | NaN Count |
|---------|------|-----|-----|-----|-----------|
| dow_sin | 0.358 | 0.514 | -0.434 | 0.975 | 0 |
| dow_cos | -0.083 | 0.775 | -0.901 | 1.000 | 0 |
| month_sin | -0.001 | 0.703 | -1.000 | 1.000 | 0 |
| month_cos | -0.018 | 0.711 | -1.000 | 1.000 | 0 |
| gap_days | 1.503 | 1.141 | 0 | 11 | 0 |
| log_return | 0.000 | 0.027 | -0.197 | 0.111 | 1 |
| return_5d | 0.002 | 0.055 | -0.398 | 0.444 | 5 |
| bb_width | 0.130 | 0.117 | 0.000 | 0.794 | 1 |
| rsi_14 | 50.29 | 10.87 | 8.62 | 92.52 | 14 |
| volume_sma_20 | 37,237 | 48,156 | 0 | 328,001 | 0 |

## Implementation Notes

### NaN Handling
- **First row**: log_return is NaN (no previous price)
- **First 5 rows**: return_5d is NaN (insufficient history)
- **First 14 rows**: rsi_14 is NaN (requires 14-day history)
- **Strategy**: Keep NaNs for LSTM to handle with masking layers

### Trading-Only Data Benefits
- **Accurate technical indicators**: All rolling windows use exactly N trading days
- **Clean momentum signals**: No dilution from non-trading periods
- **Gap information preserved**: `gap_days` captures market closure patterns

### Rolling Windows
- **20-day windows**: Exactly 20 trading days (approximately 1 month)
- **14-day RSI**: Exactly 14 trading days
- **5-day returns**: Exactly 5 trading days (one trading week)

## Feature Correlations with Close Price

Sorted by absolute correlation:
1. volume_sma_20: -0.176 (inverse relationship)
2. rsi_14: +0.118 (momentum indicator)
3. bb_width: -0.093 (volatility expansion)
4. return_5d: +0.062 (weekly momentum)
5. Others: <0.05 (weak direct correlation)

## Usage Example

```python
import pandas as pd

# Load advanced LSTM data
df = pd.read_parquet("HBEA_LSTM_advanced.parquet")

# 50 features available
print(f"Total features: {df.shape[1]}")

# Access new features
cyclical_features = ['dow_sin', 'dow_cos', 'month_sin', 'month_cos']
return_features = ['log_return', 'return_5d']
technical_features = ['bb_width', 'rsi_14', 'volume_sma_20']

# Example: Identify high volatility periods
high_volatility = df[df['bb_width'] > df['bb_width'].quantile(0.9)]

# Example: Find oversold conditions
oversold = df[df['rsi_14'] < 30]

# Example: Detect post-weekend trading days
post_weekend = df[df['gap_days'] == 3]
```

## Visualization Examples

### Cyclical Encoding
- **Day of week**: Forms a circle in (sin, cos) space, ensuring Friday-to-Monday continuity
- **Month**: Forms a circle capturing seasonal patterns without January-December discontinuity

### Gap Days Distribution
- **gap_days**: Mode at 1 (regular weekdays), secondary peak at 3 (weekends)
- Extended gaps (4-11 days) represent holiday periods

### Technical Indicators
- **RSI**: Oscillates around 50 with occasional extremes
- **Bollinger Band width**: Expands during volatile periods, contracts during calm markets

## Files

- **Notebook**: `03_Code/02_Feature_Engineering/03_add_advanced_features_LSTM.ipynb`
- **Input**: `02_Data_Processed/03_Feature_Engineered/[HBEA|GDEA]_LSTM_ready.parquet`
- **Output**: `02_Data_Processed/03_Feature_Engineered/[HBEA|GDEA]_LSTM_advanced.parquet`

## Summary

The advanced features add crucial market dynamics information:
- **Temporal patterns** without creating discontinuities
- **Market liquidity** state tracking
- **Price momentum** at multiple timescales
- **Volatility regime** detection
- **Technical signals** for overbought/oversold conditions
- **Volume patterns** for unusual activity detection

These features enable LSTM models to:
1. Learn weekly and seasonal patterns
2. Anticipate price adjustments after inactive periods
3. Detect regime changes through volatility
4. Incorporate technical trading signals
5. Identify unusual market conditions