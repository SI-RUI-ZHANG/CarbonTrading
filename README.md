# Chinese Carbon Market Price Prediction

## What This Project Does

This project predicts daily and weekly price movements in Chinese regional carbon markets (Guangdong GDEA and Hubei HBEA) using machine learning. It combines:

- **21 macroeconomic indicators** (GDP, energy consumption, commodity prices)
- **3,312 scraped policy documents** from government sources
- **Sentiment analysis** of carbon-related policies
- **LSTM neural networks** for time series prediction

## Key Results

### Model Performance Summary
| Model Type | Market | Accuracy | F1 Score | Improvement |
|------------|--------|----------|----------|-------------|
| Daily LSTM | GDEA | 51.0% | 0.481 | +19.2% with sentiment |
| Daily LSTM | HBEA | 47.6% | 0.412 | -16.5% with sentiment |
| Weekly LSTM | GDEA | 56.8% | 0.557 | +15.8% with sentiment |
| Weekly LSTM | HBEA | 52.6% | 0.434 | +39.0% with sentiment |
| Meta Model | GDEA | 56.6% | 0.596 | Best overall |

## Data Pipeline

```
Raw Data → Cleaning → Feature Engineering → Model Training → Backtesting
   ↓           ↓             ↓                    ↓              ↓
Carbon     Remove      50 technical +      LSTM with       Trading
Prices     noise       12 sentiment       walk-forward    simulation
```

## Project Structure

```
Project/
├── 01_Data_Raw/                 # Original data sources
│   ├── 01_Carbon_Markets/       # GDEA, HBEA price data (2014-2024)
│   ├── 02_Macroeconomic/        # 21 economic indicators
│   └── 03_Policy_Documents/     # 3,312 scraped documents
│
├── 02_Data_Processed/           # Cleaned and engineered features
│   ├── 03_Feature_Engineered/   # 50 base features + sentiment
│   ├── 07_Document_Scores/      # 989 unique documents scored
│   └── 12_LSTM_Weekly_Ready/    # Model-ready sequences
│
├── 03_Code/                     # Implementation
│   ├── 05_Document_Collection/  # Web scrapers (MEE, HBETS, GZETS)
│   ├── 08_Document_Scoring/     # Sentiment scoring pipeline
│   ├── 10_LSTM_Daily/          # Daily prediction models
│   ├── 11_LSTM_Weekly/         # Weekly prediction models
│   └── 12_Meta_Model/          # Error reversal meta-learning
│
└── 04_Models/                   # Trained models and results
    ├── daily/                   # Daily LSTM models
    ├── weekly/                  # Weekly LSTM models
    └── meta_reversal/          # Meta models
```

## Key Features

### 1. Document Processing
- **3,312 documents scraped** from 3 government sources
- **2,617 carbon-relevant** after filtering
- **989 unique documents** scored for sentiment
- **Regional separation**: GDEA uses Guangdong docs, HBEA uses Hubei docs

### 2. Advanced Feature Engineering
- **50 technical features**: Price momentum, volatility, market microstructure
- **12 sentiment features**: Supply/demand scores, policy strength, momentum
- **Exponential decay**: 7-day half-life for sentiment influence
- **Intelligent lags**: 1-day for daily data, 15-day for monthly macro

### 3. Robust Model Validation
- **Walk-forward validation**: 8 walks (daily), 14 walks (weekly)
- **No data leakage**: Strict temporal separation
- **Binary classification**: Up vs down/flat movement
- **Class balancing**: Handles imbalanced data

### 4. Meta-Learning Approach
- **Error reversal**: Learns from LSTM prediction errors
- **100% coverage**: No abstention, always makes predictions
- **XGBoost classifier**: Combines LSTM confidence with market features

## Running the Models

### Train Daily LSTM
```bash
cd 03_Code/10_LSTM_Daily/
python run.py --market GDEA --sentiment base      # Without sentiment
python run.py --market GDEA --sentiment sentiment # With sentiment
```

### Train Weekly LSTM
```bash
cd 03_Code/11_LSTM_Weekly/
python run.py --market HBEA --sentiment base      # Without sentiment
python run.py --market HBEA --sentiment sentiment # With sentiment
```

### Train Meta Models
```bash
cd 03_Code/12_Meta_Model/
python run.py --frequency daily --market GDEA
python run.py --frequency weekly --market HBEA
```

### Run Document Scoring
```bash
cd 03_Code/08_Document_Scoring/
python run_scoring.py  # Processes documents with GPT-4o-mini
```

## Technical Details

### LSTM Architecture
- **Input**: 60-day sequences (daily) or 30-week sequences (weekly)
- **Network**: 2-layer LSTM (64 units) → FC (64→32→1)
- **Loss**: Binary cross-entropy with class weights
- **Optimization**: Adam, learning rate 0.001
- **Early stopping**: Patience 15 epochs

### Walk-Forward Validation
- **Daily**: 700/150/200 days (train/val/test)
- **Weekly**: 180/20/30 weeks (train/val/test)
- **Total test samples**: ~1120 (daily), ~420 (weekly)

### Sentiment Features
- **Document scoring**: -150 to +150 spectrum
- **Regional separation**: Market-specific document sources
- **Aggregation**: Daily weighted by policy strength
- **Features**: Supply, demand, imbalance, momentum

## Dependencies

```python
pandas, numpy, torch, scikit-learn
beautifulsoup4, requests, pyarrow
exchange_calendars, matplotlib, xgboost
```

## Key Findings

1. **Sentiment improves weekly predictions** more than daily (+16-39% F1 improvement)
2. **GDEA market** more predictable than HBEA (56.8% vs 52.6% weekly accuracy)
3. **Meta models** achieve best overall performance (59.6% F1 for GDEA weekly)
4. **Regional document separation** critical for sentiment effectiveness

## Future Work

- Incorporate real-time news sentiment
- Add cross-market spillover effects
- Explore transformer architectures
- Implement online learning for model updates