# Regional Sentiment Analysis: Market-Specific Document Processing

## Overview

The regional sentiment system processes policy documents separately for each carbon market, recognizing that GDEA (Guangdong) and HBEA (Hubei) operate under different regulatory frameworks while sharing national-level policies. This architecture ensures each market's sentiment features reflect its unique policy environment.

## Document Source Architecture

### Three-Tier Regulatory Structure
1. **National Level (MEE)**: Ministry of Ecology and Environment
   - Affects all regional carbon markets
   - Sets national carbon policy direction
   - 37 unique scored documents

2. **Guangdong Regional (GZETS)**: Guangzhou Emissions Trading System
   - Specific to GDEA market
   - Manufacturing and export industry focus
   - 389 unique scored documents

3. **Hubei Regional (HBETS)**: Hubei Emissions Trading System
   - Specific to HBEA market
   - Heavy industry and energy sector focus
   - 563 unique scored documents

### Market-Document Mapping
```python
MARKET_SOURCES = {
    'GDEA': ['MEE', 'GZETS'],  # 426 total documents
    'HBEA': ['MEE', 'HBETS']   # 600 total documents
}
```

## Processing Pipeline

### Stage 1: Document Scoring
- **Input**: 2,617 carbon-relevant documents
- **Processing**: Direct position placement on three spectrums
  - Supply impact: -150 to +150
  - Demand impact: -150 to +150
  - Policy strength: 0 to 150
- **Output**: 989 unique documents with scores (after deduplication)

### Stage 2: Regional Filtering
Documents are filtered by market relevance:

**GDEA Pipeline**:
```
989 documents → Filter (MEE + GZETS) → 426 documents
```

**HBEA Pipeline**:
```
989 documents → Filter (MEE + HBETS) → 600 documents
```

### Stage 3: Daily Aggregation
Policy-weighted aggregation by publish date:
```python
daily_sentiment = Σ(score × policy_strength) / Σ(policy_strength)
```

### Stage 4: Feature Engineering
Each market receives 12 sentiment features:
- 5 core features (supply, demand, policy strength, averages, counts)
- 3 decay features (exponential decay with 7-day half-life)
- 2 pressure features (market imbalance indicators)
- 2 momentum features (short-term changes and news shock)

## Data Statistics

### Document Distribution
| Source | Total Docs | GDEA Docs | HBEA Docs |
|--------|-----------|-----------|-----------|
| MEE    | 37        | 37        | 37        |
| GZETS  | 389       | 389       | 0         |
| HBETS  | 563       | 0         | 563       |
| **Total** | **989** | **426**   | **600**   |

### Temporal Coverage
- **GDEA**: 378 daily observations (days with documents or decay effects)
- **HBEA**: 489 daily observations
- **Date Range**: 2013-12-19 to 2024-10-30
- **Coverage Rate**: ~15% of trading days have new documents

### Sentiment Characteristics

#### GDEA Market (Guangdong)
- **Supply**: More volatile (-120 to +100), mean 16.6
- **Demand**: Wider range (-98.6 to +100), mean 16.2
- **Policy Strength**: Higher enforcement (mean 37.5)
- **Market Pressure**: Slight downward bias (-1.1)
- **Decay Effects**: Stronger cumulative influence (mean 65.0 for supply)

#### HBEA Market (Hubei)
- **Supply**: More constrained (-30 to +120), mean 10.0
- **Demand**: Positive skew (0 to +100), mean 14.4
- **Policy Strength**: Lower enforcement (mean 19.7)
- **Market Pressure**: Slight upward bias (+1.3)
- **Decay Effects**: Moderate cumulative influence (mean 33.0 for supply)

## Implementation Details

### File Structure
```
02_Data_Processed/
├── 07_Document_Scores/
│   ├── document_scores.parquet         # 989 unique scored documents
│   ├── batch_scores/                   # Intermediate batch files
│   └── checkpoint.json                 # Processing checkpoint
├── 09_Sentiment_Engineered/
│   ├── GDEA_sentiment_daily.parquet    # GDEA sentiment features
│   ├── HBEA_sentiment_daily.parquet    # HBEA sentiment features
│   ├── feature_statistics_GDEA.json    # GDEA feature statistics
│   └── feature_statistics_HBEA.json    # HBEA feature statistics
└── 03_Feature_Engineered/
    ├── GDEA_LSTM_with_sentiment.parquet # GDEA + sentiment
    └── HBEA_LSTM_with_sentiment.parquet # HBEA + sentiment
```

### Processing Scripts
1. **`08_Document_Scoring/score_documents.py`**
   - Scores all documents using GPT-4
   - Resilient batch processing with checkpoints
   - Automatic deduplication

2. **`09_Sentiment_Features/02_engineer_sentiment_features.py`**
   - Implements regional separation
   - Calculates decay and momentum features
   - Generates market-specific outputs

3. **`09_Sentiment_Features/03_merge_with_market.py`**
   - Merges sentiment with carbon price data
   - Handles non-trading days
   - Creates LSTM-ready datasets

## Key Design Decisions

### Why Regional Separation?
1. **Regulatory Independence**: Each market has its own trading rules and compliance requirements
2. **Industry Focus**: GDEA serves manufacturing, HBEA serves heavy industry
3. **Policy Effectiveness**: Regional policies have stronger local impact than in other regions
4. **Market Dynamics**: Different supply-demand characteristics require separate sentiment tracking

### Why Include MEE for Both Markets?
1. **National Coordination**: MEE sets overall carbon reduction targets
2. **Policy Harmonization**: National policies ensure market compatibility
3. **Systemic Shocks**: Major national policies affect all markets simultaneously
4. **Baseline Sentiment**: Provides common reference across markets

### Why Exponential Decay?
Markets have memory but with diminishing influence:
- **Day 0**: 100% influence (immediate reaction)
- **Day 7**: 50% influence (half-life)
- **Day 14**: 25% influence (fading memory)
- **Day 30**: 6.25% influence (background noise)

This captures both immediate policy shocks and lingering market effects.

### Why Policy-Weighted Aggregation?
Not all documents are equal:
- **Mandatory regulations** (strength >67): Drive immediate market action
- **Binding guidelines** (strength 33-67): Influence medium-term planning
- **Informational notices** (strength <33): Minimal market impact

Weighting by policy strength ensures market-moving documents dominate sentiment.

## Integration with LSTM Models

### Feature Enhancement
Original LSTM features: 49 (technical + macro indicators)
With sentiment features: 61 (49 original + 12 sentiment)

### Model Training
```python
# Update data source in config
GDEA_DATA = "02_Data_Processed/03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet"
HBEA_DATA = "02_Data_Processed/03_Feature_Engineered/HBEA_LSTM_with_sentiment.parquet"

# Features automatically detected
num_features = data.shape[1] - 1  # Excludes target
```

### Expected Impact
Sentiment features capture policy-driven market movements that technical indicators miss:
- **Supply shocks**: New quota allocations or restrictions
- **Demand changes**: Industry compliance requirements
- **Regulatory shifts**: Enforcement intensity changes
- **Market confidence**: Policy consistency and clarity

## Performance Considerations

### Processing Efficiency
- Document scoring: ~7-10 minutes for 2,617 documents
- Regional filtering: <1 second
- Feature engineering: <5 seconds per market
- Total pipeline: ~10 minutes end-to-end

### Memory Usage
- Document scores: 989 records × 6 columns = ~240KB
- Daily features: ~500 days × 12 features × 2 markets = ~96KB
- Merged datasets: ~61 features × ~3000 days × 2 markets = ~2.9MB

### Resilience Features
1. **Checkpoint System**: Resume from interruption without data loss
2. **Batch Processing**: 50 documents per batch for API stability
3. **Deduplication**: Automatic removal of duplicate documents
4. **Persistent Numbering**: Batch files never overwritten across runs

## Future Enhancements

### Potential Improvements
1. **Cross-Market Influence**: Model policy spillover between markets
2. **Sector-Specific Sentiment**: Separate sentiment for different industries
3. **Event Detection**: Identify policy shocks and regime changes
4. **Adaptive Decay**: Learn optimal decay rates from data

### Scalability
The architecture easily extends to:
- National carbon market (when launched)
- Additional regional markets (Beijing, Shanghai, etc.)
- International markets (EU ETS integration)
- Other commodity markets with policy influence