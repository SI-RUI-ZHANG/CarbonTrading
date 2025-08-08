# LSTM Data Preparation Pipeline

## Overview

This document describes the complete LSTM data preparation pipeline, which transforms raw carbon market data through multiple stages of feature engineering to create sequence-ready arrays for model training.

**Pipeline Stages:**
1. **Data Age Features** - Track freshness of macroeconomic indicators
2. **Advanced Features** - Add technical indicators and market state features  
3. **Sequence Generation** - Create 3D arrays with sliding windows for LSTM input

**Final Output:** Market-specific numpy arrays and scalers ready for LSTM training

---

## Stage 1: Data Age Features

### Purpose
Add "data age" features to track the freshness of macroeconomic indicators. Since macro data is forward-filled between observations, LSTM models need to distinguish between actual data persistence and staleness from forward-filling.

### Implementation

#### Data Age Calculation
For each macroeconomic column:
- Age = 0 when value changes (fresh observation)
- Age increments daily while value remains constant (forward-filled)

```python
def add_data_age(df, column):
    value_changed = df[column].ne(df[column].shift())
    age = (~value_changed).groupby(value_changed.cumsum()).cumsum()
    return age
```

#### Column Selection
**Skip (always fresh):**
- Carbon market: `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`, `is_open`, `is_quiet`, `has_trade`

**Add age tracking:**
- All macro indicators (columns ending in `_1` or `_15`)

#### Age Patterns by Frequency

| Indicator Type | Typical Max Age | Reset Frequency |
|----------------|-----------------|-----------------|
| Daily (FX, futures) | 3-5 days | Weekends/holidays |
| Monthly (CPI, PMI) | 28-31 days | Monthly |
| Quarterly (GDP) | 89-92 days | Quarterly |

### Output
- **Files**: `[HBEA|GDEA]_LSTM_ready.parquet`
- **Location**: `02_Data_Processed/03_Feature_Engineered/`
- **Features**: 42 total (25 original + 17 age features)

---

## Stage 2: Advanced Technical Features

### Purpose
Add 10 advanced technical and temporal features for enhanced pattern recognition.

### Feature Categories

#### Cyclical Time (4 features)
- `dow_sin`, `dow_cos`: Day of week encoding
- `month_sin`, `month_cos`: Month encoding
- **Benefit**: Preserves continuity (Sunday→Monday, December→January)

#### Market State (1 feature)
- `days_since_trade`: Consecutive days without trading
- **Pattern**: 0 on trading days, 1-2 for weekends, 3-7 for holidays
- **Max observed**: 59 days (COVID-19 disruption)

#### Price Returns (2 features)
- `log_return`: Daily logarithmic returns
- `return_5d`: 5-day percentage returns
- **Note**: Returns are 0 on non-trading days (forward-filled prices)

#### Volatility (1 feature)
- `bb_width`: Normalized Bollinger Band width
- **Formula**: (Upper Band - Lower Band) / SMA(20)

#### Momentum (1 feature)
- `rsi_14`: 14-day Relative Strength Index
- **Range**: [0, 100] (>70 overbought, <30 oversold)

#### Volume (1 feature)
- `volume_sma_20`: 20-day volume moving average

### Output
- **Files**: `[HBEA|GDEA]_LSTM_advanced.parquet`
- **Location**: `02_Data_Processed/03_Feature_Engineered/`
- **Features**: 52 total (42 + 10 advanced features)

---

## Stage 3: Sequence Generation

### Purpose
Transform 2D feature DataFrames into 3D sequence arrays suitable for LSTM input, with proper train/validation/test splits and scaling.

### Configuration

```python
# Key Parameters
SEQUENCE_LENGTH = 60      # 60-day lookback window
TRAIN_END_DATE = '2020-12-31'
VAL_END_DATE = '2022-12-31'
TARGET_COLUMN = 'log_return'  # Next-day return prediction
```

### Processing Steps

#### 1. Data Cleaning
- Load `[MARKET]_LSTM_advanced.parquet`
- Remove rows with NaN values (initial technical indicators)
- Result: ~3,977 clean rows from 4,027 original

#### 2. Chronological Splitting
Strict temporal splits to prevent data leakage:

| Split | Date Range | Samples (HBEA) | Samples (GDEA) |
|-------|------------|----------------|----------------|
| Train | 2014 → 2020-12-31 | 2,366 | 2,306 |
| Val | 2021-01-01 → 2022-12-31 | 670 | 670 |
| Test | 2023-01-01 → 2025+ | 761 | 761 |

#### 3. Feature Scaling
- **Scaler**: MinMaxScaler with range [0, 1]
- **Critical**: Fit on training features ONLY
- **Features**: All 51 columns except target
- **Target**: Kept in original scale (log returns)

#### 4. Sliding Window Sequences
Create overlapping sequences with 60-day lookback:

```python
def create_sequences(features, targets, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])  # Past 60 days
        y.append(targets[i])                      # Next day target
    return np.array(X), np.array(y)
```

**Output Shapes:**
- X arrays: (samples, 60, 51) - 3D for LSTM
- y arrays: (samples,) - 1D targets

### Output Files

All files saved to `02_Data_Processed/04_LSTM_Ready/`:

| File | Description | Shape |
|------|-------------|-------|
| `[MARKET]_feature_scaler.pkl` | Fitted MinMaxScaler | (51 features) |
| `[MARKET]_X_train.npy` | Training sequences | (2366, 60, 51) |
| `[MARKET]_y_train.npy` | Training targets | (2366,) |
| `[MARKET]_X_val.npy` | Validation sequences | (670, 60, 51) |
| `[MARKET]_y_val.npy` | Validation targets | (670,) |
| `[MARKET]_X_test.npy` | Test sequences | (761, 60, 51) |
| `[MARKET]_y_test.npy` | Test targets | (761,) |

*Note: [MARKET] = HBEA or GDEA*

### Data Distribution
- **Train**: ~62% of total samples
- **Validation**: ~18% of total samples  
- **Test**: ~20% of total samples
- **Total memory**: ~88 MB per market

---

## Implementation Files

### Notebooks (Feature Engineering)
1. `03_Code/02_Feature_Engineering/02_add_data_age_LSTM.ipynb` - Data age features
2. `03_Code/02_Feature_Engineering/03_add_advanced_features_LSTM.ipynb` - Technical features

### Scripts (Sequence Generation)
- `03_Code/04_LSTM_Model/data_preparation.py` - Complete pipeline script

### Usage Example

```python
# Process HBEA market
MARKET = 'HBEA'
python data_preparation.py

# Process GDEA market  
MARKET = 'GDEA'
python data_preparation.py

# Load prepared data for training
import numpy as np
import joblib

X_train = np.load('02_Data_Processed/04_LSTM_Ready/HBEA_X_train.npy')
y_train = np.load('02_Data_Processed/04_LSTM_Ready/HBEA_y_train.npy')
scaler = joblib.load('02_Data_Processed/04_LSTM_Ready/HBEA_feature_scaler.pkl')

print(f"Training samples: {X_train.shape[0]}")
print(f"Sequence length: {X_train.shape[1]}")
print(f"Features per timestep: {X_train.shape[2]}")
```

---

## Key Design Decisions

### Why These Choices?

1. **60-day lookback**: Captures quarterly patterns while maintaining sufficient training samples
2. **Log returns as target**: Better statistical properties than raw returns
3. **MinMaxScaler [0,1]**: Optimal for LSTM gradient stability
4. **Train-only scaler fitting**: Prevents data leakage
5. **Chronological splits**: Respects temporal nature of financial data
6. **Market-specific files**: Enables independent model training for each market

### Data Quality Considerations

- **NaN handling**: Dropped initial rows where technical indicators lack history
- **Forward-filled prices**: Returns are 0 on non-trading days
- **Age features**: Critical for distinguishing fresh vs stale macro data
- **Validation range**: Some features exceed [0,1] in val/test due to market evolution

---

## Summary

The LSTM data preparation pipeline transforms raw carbon market data through three stages:

1. **Data age tracking** (42 features) → Freshness awareness
2. **Technical features** (52 features) → Market dynamics
3. **Sequence generation** (3D arrays) → LSTM-ready format

The pipeline ensures:
- No data leakage through proper chronological splits
- Market-specific processing for HBEA and GDEA
- Reproducible results with saved scalers
- Memory-efficient numpy array storage

**Next Step**: Use these prepared arrays to train LSTM models for carbon price prediction.