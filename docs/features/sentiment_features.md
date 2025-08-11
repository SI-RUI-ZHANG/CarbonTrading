# Sentiment Feature Engineering

## Overview
This module engineers advanced sentiment features from policy document scores to capture market sentiment dynamics for carbon price prediction.

## Data Flow
```
Document Scores (07_Document_Scores/)
    ↓
Daily Aggregation (policy-weighted)
    ↓
Decay Calculation (7-day half-life)
    ↓
Market Pressure & Momentum
    ↓
Sentiment Features (09_Sentiment_Engineered/)
    ↓
Merge with Market Data
    ↓
Final LSTM Input (03_Feature_Engineered/)
```

## Features Created

### 1. Core Sentiment Features
- **sentiment_supply**: Policy-weighted daily aggregate of supply scores
- **sentiment_demand**: Policy-weighted daily aggregate of demand scores
- **max_policy**: Maximum policy strength score for the day
- **avg_policy**: Average policy strength score for the day
- **doc_count**: Number of documents published that day

### 2. Decay Features (7-day half-life)
- **supply_decayed**: Cumulative decayed supply sentiment from all past documents
- **demand_decayed**: Cumulative decayed demand sentiment from all past documents
- **policy_decayed**: Cumulative decayed policy strength from all past documents

Formula: `decay_factor = 0.5^(days_elapsed / 7)`

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

## Implementation Files

### 1. `engineer_sentiment_features.py`
Main feature engineering pipeline that:
- Loads raw document scores
- Applies policy-weighted daily aggregation
- Calculates exponential decay features
- Adds market pressure and momentum indicators
- Saves to `09_Sentiment_Engineered/`

### 2. `merge_with_market.py`
Combines sentiment with market data:
- Merges sentiment features with GDEA/HBEA market data
- Handles non-trading days with forward-fill
- Creates final LSTM-ready datasets

## Usage

```bash
# Step 1: Generate sentiment features
cd 03_Code/09_Sentiment_Features/
python engineer_sentiment_features.py

# Step 2: Merge with market data
python merge_with_market.py

# Output files created:
# - 02_Data_Processed/09_Sentiment_Engineered/sentiment_daily_features.parquet
# - 02_Data_Processed/03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet
# - 02_Data_Processed/03_Feature_Engineered/HBEA_LSTM_with_sentiment.parquet
```

## Feature Rationale

### Why Policy-Weighted Aggregation?
High policy-strength documents (regulations, mandatory rules) have more market impact than low-strength ones (announcements, information). Weighting by policy strength ensures important documents dominate the daily sentiment.

### Why 7-Day Decay?
Markets digest information quickly. A 7-day half-life means:
- Day 0: 100% influence
- Day 7: 50% influence
- Day 14: 25% influence
- Day 30: ~6% influence

This captures how markets "forget" old news while still considering recent history.

### Why Market Pressure?
The fundamental driver of prices is supply-demand imbalance. Market pressure captures this directly and weights it by policy importance.

### Why Momentum?
Short-term trends matter in financial markets. Momentum features capture whether sentiment is accelerating or decelerating.

## Integration with LSTM

To use these features in the LSTM model:

1. Update `04_LSTM_Model/data_preparation.py`:
```python
# Change data source
data_path = "02_Data_Processed/03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet"
```

2. The new features are automatically included as the merged files contain all original features plus sentiment features.

## Feature Statistics

The pipeline generates `feature_statistics.json` with mean, std, min, max for each feature, useful for:
- Normalization
- Outlier detection
- Feature scaling in LSTM