# Regional Sentiment Feature Engineering

## Overview
This module generates market-specific sentiment features from policy documents using regional document separation. GDEA and HBEA markets process different document sources to capture their unique regulatory environments while sharing national-level policies.

## Regional Architecture

### Market-Source Separation
```python
MARKET_SOURCES = {
    'GDEA': ['MEE', 'GZETS'],  # Guangdong: National + Guangdong-specific
    'HBEA': ['MEE', 'HBETS']   # Hubei: National + Hubei-specific
}
```

### Document Distribution
- **GDEA Market**: 426 documents → 378 daily observations
  - MEE (Ministry of Ecology): National policies affecting all markets
  - GZETS (Guangzhou Emissions Trading): Guangdong-specific policies
- **HBEA Market**: 600 documents → 489 daily observations
  - MEE: Same national policies
  - HBETS (Hubei Emissions Trading): Hubei-specific policies

## Data Flow
```
Document Scores (989 unique documents)
    ↓
Regional Filtering (GDEA: MEE+GZETS, HBEA: MEE+HBETS)
    ↓
Daily Aggregation (policy-weighted per market)
    ↓
Exponential Decay (7-day half-life)
    ↓
Market Pressure & Momentum
    ↓
Regional Sentiment Features
    ↓
Merge with Market Data
    ↓
LSTM-Ready Datasets with Sentiment
```

## Features Created (12 per market)

### 1. Core Sentiment Features
- **sentiment_supply**: Policy-weighted daily aggregate of supply scores (-150 to +150)
- **sentiment_demand**: Policy-weighted daily aggregate of demand scores (-150 to +150)
- **max_policy**: Maximum policy strength score for the day (0 to 150)
- **avg_policy**: Average policy strength score for the day
- **doc_count**: Number of documents published that day

### 2. Decay Features (7-day half-life)
- **supply_decayed**: Cumulative decayed supply sentiment from all past documents
- **demand_decayed**: Cumulative decayed demand sentiment from all past documents
- **policy_decayed**: Cumulative decayed policy strength from all past documents

Decay formula: `decay_factor = 0.5^(days_elapsed / 7)`

### 3. Market Pressure Features
- **market_pressure**: Supply-demand imbalance weighted by policy strength
  - Formula: `(demand - supply) * (max_policy / 100)`
- **pressure_magnitude**: Total market activity indicator
  - Formula: `(|demand| + |supply|) * (max_policy / 100)`

### 4. Momentum Features
- **news_shock**: Deviation from normal document flow
  - Formula: `doc_count / rolling_mean_7days`
- **pressure_momentum**: 3-day change in market pressure
- **supply_momentum**: 3-day change in supply sentiment
- **demand_momentum**: 3-day change in demand sentiment

## Feature Statistics

### GDEA Market (MEE + GZETS)
Key characteristics from 378 daily observations:
- **Supply Sentiment**: Mean 16.6, ranges from -120 to +100
- **Demand Sentiment**: Mean 16.2, ranges from -98.6 to +100
- **Max Policy Strength**: Mean 37.5, median 33 (voluntary-guided range)
- **Supply Decayed**: Mean 65.0, shows cumulative policy influence
- **Market Pressure**: Mean -1.1, slight downward price pressure
- **News Shock**: Mean 0.93, relatively stable document flow

### HBEA Market (MEE + HBETS)
Key characteristics from 489 daily observations:
- **Supply Sentiment**: Mean 10.0, ranges from -30 to +120
- **Demand Sentiment**: Mean 14.4, ranges from 0 to +100
- **Max Policy Strength**: Mean 19.7, median 33 (lower enforcement)
- **Supply Decayed**: Mean 33.0, lower cumulative influence than GDEA
- **Market Pressure**: Mean +1.3, slight upward price pressure
- **News Shock**: Mean 0.92, stable document flow

## Implementation Files

### 1. `02_engineer_sentiment_features.py`
Main regional sentiment pipeline:
- Loads 989 unique document scores
- Filters documents by market-specific sources
- Applies policy-weighted daily aggregation per market
- Calculates exponential decay features
- Adds market pressure and momentum indicators
- Saves separate files for GDEA and HBEA

### 2. `03_merge_with_market.py`
Combines regional sentiment with market data:
- Merges GDEA sentiment with GDEA market data
- Merges HBEA sentiment with HBEA market data
- Handles non-trading days with forward-fill
- Creates final LSTM-ready datasets with all features

## Usage

```bash
# Generate regional sentiment features
cd 03_Code/09_Sentiment_Features/
python 02_engineer_sentiment_features.py

# Merge with market data
python 03_merge_with_market.py

# Output files created:
# - 02_Data_Processed/09_Sentiment_Engineered/GDEA_sentiment_daily.parquet
# - 02_Data_Processed/09_Sentiment_Engineered/HBEA_sentiment_daily.parquet
# - 02_Data_Processed/09_Sentiment_Engineered/feature_statistics_GDEA.json
# - 02_Data_Processed/09_Sentiment_Engineered/feature_statistics_HBEA.json
# - 02_Data_Processed/03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet
# - 02_Data_Processed/03_Feature_Engineered/HBEA_LSTM_with_sentiment.parquet
```

## Feature Engineering Rationale

### Why Regional Separation?
Different carbon markets have distinct regulatory environments:
- Guangdong focuses on manufacturing and export industries
- Hubei emphasizes heavy industry and energy sectors
- Regional policies reflect these different priorities
- Shared national policies (MEE) affect both markets

### Why Policy-Weighted Aggregation?
Documents vary in regulatory importance:
- Mandatory regulations (high policy strength) drive immediate market responses
- Informational notices (low policy strength) have minimal impact
- Weighting ensures market-moving documents dominate daily sentiment

### Why 7-Day Decay?
Markets digest policy information at different rates:
- Immediate reaction: Day 0-3 (>70% influence retained)
- Fading memory: Day 7 (50% influence)
- Background noise: Day 30 (~6% influence)
- Captures both immediate shocks and lingering effects

### Why Market Pressure?
Price fundamentals come from supply-demand dynamics:
- Positive pressure: Demand exceeds supply → upward price pressure
- Negative pressure: Supply exceeds demand → downward price pressure
- Policy strength amplifies or dampens these pressures

## Integration with LSTM Models

The regional sentiment features seamlessly integrate with existing LSTM infrastructure:

1. **Data Loading**: Update paths to sentiment-enhanced datasets
```python
# For GDEA model
data_path = "02_Data_Processed/03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet"

# For HBEA model
data_path = "02_Data_Processed/03_Feature_Engineered/HBEA_LSTM_with_sentiment.parquet"
```

2. **Feature Count**: Automatically detected (now 61 features including 12 sentiment)

3. **Model Architecture**: No changes needed - dynamic input size handles new features

## Key Insights

### Market Comparison
- **GDEA**: Higher policy strength, more volatile sentiment, stronger decay effects
- **HBEA**: Lower policy strength, more stable sentiment, moderate decay effects
- **Common Pattern**: Both markets show news shock stability around 0.9

### Sentiment Dynamics
- Supply and demand sentiments often move together (policy coordination)
- Market pressure oscillates around zero (market equilibrium tendency)
- Momentum features capture short-term sentiment shifts

### Data Quality
- 100% coverage: All trading days have sentiment features (zero-filled for no-document days)
- Regional integrity: Each market only sees its relevant documents
- Temporal consistency: Decay ensures past policies influence current sentiment