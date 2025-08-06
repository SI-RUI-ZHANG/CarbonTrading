# LSTM Advanced Features Documentation

## Overview

This document details the 10 advanced technical and temporal features added to enhance LSTM modeling capabilities for carbon price prediction.

**Purpose**: Capture market dynamics, temporal patterns, and technical indicators  
**Input**: LSTM-ready data with 42 features (including age tracking)  
**Output**: LSTM-advanced data with 52 features  
**Implementation**: `03_Code/02_Feature_Engineering/03_add_advanced_features_LSTM.ipynb`

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

### 2. Market State Features (1 feature)

#### days_since_trade
Counts consecutive days without trading activity.

```python
trade_event = df['has_trade'].astype(int)
trade_groups = trade_event.cumsum()
days_counter = df.groupby(trade_groups).cumcount()
days_since_trade = days_counter * (~df['has_trade']).astype(int)
```

**Distribution**:
- 0: Trading day (65% of days)
- 1-2: Typical weekend
- 3-7: Holiday periods
- Max observed: 59 days (COVID-19 market disruption, March 2020)

**Purpose**: Captures market liquidity state and predicts potential price gaps after inactive periods.

### 3. Price Returns (2 features)

#### log_return
Daily logarithmic returns for better statistical properties.

```python
log_return = np.log(df['close'] / df['close'].shift(1))
```

- Mean: 0.000128 (slight positive drift)
- Std: 0.022366 (2.24% daily volatility)
- **Note**: Returns are 0 on non-trading days due to forward-filled prices

#### return_5d
5-day (weekly) percentage returns for medium-term momentum.

```python
return_5d = df['close'].pct_change(5)
```

- Mean: 0.001732 (0.17% weekly return)
- Captures weekly momentum patterns

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

- Mean: 0.1085 (10.85% average band width)
- Std: 0.1031
- **Interpretation**: Higher values indicate increased volatility/uncertainty

### 5. Technical Momentum (1 feature)

#### rsi_14
14-day Relative Strength Index using Wilder's smoothing.

```python
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta.clip(upper=0))
avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
rs = avg_gain / avg_loss
rsi_14 = 100 - (100 / (1 + rs))
```

- Range: [0, 97.3]
- Mean: 50.34 (neutral market)
- **Thresholds**: >70 = overbought, <30 = oversold

### 6. Volume Analysis (1 feature)

#### volume_sma_20
20-day simple moving average of trading volume.

```python
volume_sma_20 = df['volume_tons'].rolling(20, min_periods=1).mean()
```

- Mean: 24,745 tons
- Ratio to actual volume: 1.00 (well-calibrated)
- **Purpose**: Baseline for detecting unusual trading activity

## Feature Statistics Summary

| Feature | Mean | Std | Min | Max | NaN Count |
|---------|------|-----|-----|-----|-----------|
| dow_sin | 0.001 | 0.707 | -0.975 | 0.975 | 0 |
| dow_cos | 0.037 | 0.756 | -0.901 | 1.000 | 0 |
| month_sin | -0.003 | 0.707 | -1.000 | 1.000 | 0 |
| month_cos | 0.010 | 0.708 | -1.000 | 1.000 | 0 |
| days_since_trade | 2.337 | 5.483 | 0 | 59 | 0 |
| log_return | 0.000128 | 0.022 | -0.300 | 0.335 | 1 |
| return_5d | 0.002 | 0.050 | -0.451 | 0.533 | 5 |
| bb_width | 0.109 | 0.103 | 0.004 | 1.251 | 1 |
| rsi_14 | 50.34 | 14.32 | 0.00 | 97.30 | 14 |
| volume_sma_20 | 24,745 | 16,625 | 75 | 111,445 | 0 |

## Implementation Notes

### NaN Handling
- **First row**: log_return is NaN (no previous price)
- **First 5 rows**: return_5d is NaN (insufficient history)
- **First 14 rows**: rsi_14 is NaN (requires 14-day history)
- **Strategy**: Keep NaNs for LSTM to handle with masking layers

### Edge Cases
- **Non-trading days**: Returns are 0 due to forward-filled prices
- **Market reopening**: days_since_trade resets to 0, potential for price gaps
- **Extreme volatility**: bb_width can spike during crisis periods

### Rolling Windows
- **20-day windows**: Approximately 1 month of trading days
- **14-day RSI**: Standard technical analysis period
- **5-day returns**: One trading week

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

# 52 features available
print(f"Total features: {df.shape[1]}")

# Access new features
cyclical_features = ['dow_sin', 'dow_cos', 'month_sin', 'month_cos']
market_features = ['days_since_trade']
return_features = ['log_return', 'return_5d']
technical_features = ['bb_width', 'rsi_14', 'volume_sma_20']

# Example: Identify high volatility periods
high_volatility = df[df['bb_width'] > df['bb_width'].quantile(0.9)]

# Example: Find oversold conditions
oversold = df[df['rsi_14'] < 30]
```

## Visualization Examples

### Cyclical Encoding
- **Day of week**: Forms a circle in (sin, cos) space, ensuring Friday-to-Monday continuity
- **Month**: Forms a circle capturing seasonal patterns without January-December discontinuity

### Market State
- **days_since_trade distribution**: Heavy right tail with mode at 0 (trading days)
- COVID-19 period shows exceptional 59-day gap

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