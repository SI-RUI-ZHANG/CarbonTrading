## Project Overview

Carbon trading research project analyzing Chinese regional carbon markets (Guangdong GDEA and Hubei HBEA). Predicts daily price movements using LSTM models with macroeconomic features. Includes 2,367 scraped policy documents for future NLP integration.

## Directory Structure

- **01_Data_Raw/**: Raw data sources
  - `01_Carbon_Markets/`: GDEA, HBEA price data
  - `02_Macroeconomic_Indicators/`: 21 economic indicators
  - `03_Policy_Documents/`: MEE (573), HBETS (684), GZETS (1,110) scraped docs
- **02_Data_Processed/**:
  - `01_Carbon_Markets/01_Regional/`: Cleaned carbon price data
  - `02_Macroeconomic_Indicators/`: Forward-filled, interpolated, daily-aligned versions
  - `03_Feature_Engineered/`: GDEA/HBEA_daily_with_macro.parquet, *_LSTM_advanced.parquet
  - `04_LSTM_Ready/`: Model-ready datasets
- **03_Code/**:
  - `01_Data_Cleaning/`: Carbon markets and macro indicators processing
  - `02_Feature_Engineering/`: Lagged macro features join
  - `03_Base_Models/`: Buy&Hold, MA, MACD, RSI strategies
  - `04_LSTM_Model/`: Binary classification model (up/down-flat)
  - `05_Web_Scraping/`: MEE, HBETS, GZETS scrapers
- **04_Models/**: Timestamped experiment folders with trained models
- **docs/**: analysis/, data/, models/, scrapers/ documentation

## Key Commands

### Python Environment
```bash
# Core dependencies (no requirements.txt yet)
pip install pandas numpy matplotlib exchange_calendars pyarrow torch scikit-learn beautifulsoup4 requests

# Run notebooks in sequence (numbered files)
jupyter notebook
```

### Model Training
```bash
# Train LSTM model
cd 03_Code/04_LSTM_Model/
python model_training.py  # Creates timestamped folder in 04_Models/
```

### Web Scraping
```bash
# Re-run scrapers (incremental, skips existing)
cd 03_Code/05_Web_Scraping/
python 01_scrape_mee.py
python 04_scrape_hbets.py
python 05_scrape_gzets.py
```

## Architecture & Design Patterns

### Data Processing Pipeline
1. **Trading Calendar Alignment**: Uses Shanghai Stock Exchange (XSHG) calendar for all data
2. **Missing Value Handling**: Forward-fill → Linear interpolation → Daily alignment
3. **Lag Strategy**: 1-day lag for daily data, 15-day lag for monthly/quarterly macro indicators
4. **File Naming Convention**:
   - `*_processed.csv/.parquet` - Basic cleaned data
   - `*_forward_filled.csv/.parquet` - Forward-filled missing values
   - `*_interpolated.csv/.parquet` - Linearly interpolated
   - `*_ffill_daily.parquet` - Forward-filled and daily aligned

### Trading Strategy Framework (`03_Code/03_Base_Models/`)
- **Abstract Base**: `a_Strategy.py` - Strategy ABC with `run()` method returning (signal, NAV)
- **NAV Simulation**: `simulate_nav()` - Converts signals to NAV with proper lag (t+1 execution)
- **Evaluation**: `a_Evaluation.py` - Performance metrics (CAGR, Sharpe, volatility)
- **Comparison**: `b_Strategy_Performance.py` - Multi-strategy backtesting and visualization

### LSTM Model Architecture (`03_Code/04_LSTM_Model/`)
- **Config-driven**: `config.py` - Centralized configuration with experiment tracking
- **Binary Classification**: Predicts up vs down/flat price movement
- **Dynamic Input Size**: Automatically detects feature count from data
- **Experiment Tracking**: Each run creates unique folder in `04_Models/` with timestamp

### Web Scraping Infrastructure (`03_Code/05_Web_Scraping/`)
- **Parallel Processing**: All scrapers use ThreadPoolExecutor (MEE: 5 workers, others: 10)
- **Storage**: Individual JSONs per document + `_all_documents.jsonl` compilation
- **Progress Tracking**: `progress.json` for incremental scraping
- **Data Quality**: Multiple date extraction methods (JavaScript vars, Chinese dates, URL patterns)

## Important Constants

```python
# Trading Strategy
INIT_CAPITAL = 1000000  # Initial capital for backtesting
TRADING_DAYS = 252      # Trading days per year for annualization

# LSTM Model
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Web Scraping
MAX_WORKERS = 10  # HBETS/GZETS (MEE uses 5)
RETRY_ATTEMPTS = 3
```

## Data Quality

- **Trading Calendar**: XSHG (Shanghai Stock Exchange) for all alignments
- **Missing Values**: Forward-fill → Linear interpolation → Daily alignment
- **Scraped Documents**: 100% have valid publish_date and clean content after fixes
- **Lag Structure**: 1-day for daily data, 15-day for monthly/quarterly macro data

## Implementation Status

### Completed Components
- Carbon market data: GDEA (2014-2024), HBEA (2014-2024) cleaned and aligned
- Macroeconomic indicators: 21 indicators with intelligent lag (1-day daily, 15-day monthly)
- Feature engineering: 100+ technical indicators, rolling statistics, market microstructure
- Baseline strategies: All implemented with NAV simulation and evaluation
- LSTM models: GDEA (59.8% accuracy, 56.3% F1), HBEA (trained)
- Web scraping: 2,367 documents with 100% date extraction and content cleaning
  - MEE: 573 docs (89 Decrees, 484 Notices)
  - HBETS: 684 docs (Center Dynamics)
  - GZETS: 1,110 docs (342 Trading, 438 Center, 330 Provincial)

## Key Context

- **Data Pipeline Order**: Clean → Forward-fill → Interpolate → Daily-align → Feature engineer
- **Model Performance**: GDEA 59.8% accuracy on price direction, better than random (50%)
- **Scraper Output**: Each doc has doc_id, url, title, content, publish_date, section, source
- **Critical Files**:
  - `02_Data_Processed/03_Feature_Engineered/*_LSTM_advanced.parquet` - Model input
  - `04_Models/*/metrics.json` - Model performance
  - `01_Data_Raw/03_Policy_Documents/*/_all_documents.jsonl` - Scraped docs