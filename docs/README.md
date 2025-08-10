# Carbon Trading Project Documentation

Comprehensive documentation for the Chinese regional carbon markets analysis project.

## 📊 Data Pipeline
- [Data Sources Overview](01_data_pipeline/01_data_sources.md) - All data sources and their characteristics
- [Carbon Markets Processing](01_data_pipeline/02_carbon_markets.md) - GDEA/HBEA data cleaning and alignment
- [Macroeconomic Indicators](01_data_pipeline/03_macroeconomic.md) - Processing 21 economic indicators
- [Lag Structure Strategy](01_data_pipeline/04_lag_structure.md) - Intelligent lag for different data frequencies
- [Feature Engineering](01_data_pipeline/05_feature_engineering.md) - Advanced LSTM features creation

## 🤖 Models
- [Baseline Strategies](02_models/01_baseline_strategies.md) - Buy&Hold, MA, MACD, RSI implementations
- [LSTM Architecture](02_models/02_lstm_architecture.md) - Binary classification model design
- [LSTM Features](02_models/03_lstm_features.md) - 50 advanced features for prediction
- [Model Evaluation](02_models/04_model_evaluation.md) - Performance metrics and comparison

## 📄 Document Processing
- [Web Scraping](03_document_processing/01_web_scraping.md) - Scraping MEE, HBETS, GZETS sources
- [Document Cleaning](03_document_processing/02_document_cleaning.md) - Removing navigation and artifacts
- [Carbon Filtering](03_document_processing/03_carbon_filtering.md) - Identifying carbon-relevant documents
- **[Anchor Selection](03_document_processing/04_anchor_selection.md)** - MapReduce exemplar selection
- **[Document Scoring](03_document_processing/05_document_scoring.md)** - Spectrum positioning system

## 📈 Analysis
- [Data Publication Lag](04_analysis/01_data_lag_analysis.md) - Analysis of data availability delays
- [Model Performance](04_analysis/02_model_performance.md) - Comparative results across strategies
- [Score Analysis](04_analysis/03_score_analysis.md) - Document scoring insights

## 🔧 API Reference
- [Carbon Markets API](05_api_reference/carbon_markets_api.md)
- [Document Processing API](05_api_reference/document_processing_api.md)
- [Model API](05_api_reference/model_api.md)

---

## Key Innovations

### 1. Intelligent Lag Structure
Different lag strategies for daily (1-day) vs monthly/quarterly (15-day) data, respecting information availability in real markets.

### 2. MapReduce Anchor Selection
Scalable selection of 12 exemplar documents from 2,617 using parallel processing and binary tournament merging.

### 3. Spectrum Positioning
Documents scored on bidirectional spectrums (-100 to +100) for supply/demand, capturing directional policy impacts.

### 4. Smart Rate Limiting
Adaptive API throttling that automatically recovers from rate limits while maximizing throughput.

## Quick Start

1. **Data Pipeline**: Start with carbon markets processing to understand the foundation
2. **Document System**: Read anchor selection → document scoring for the NLP pipeline
3. **Models**: Review baseline strategies before diving into LSTM architecture

## Project Statistics

- **Data Coverage**: 2014-2024 (10 years)
- **Documents Processed**: 2,617 carbon-relevant policies
- **Features Generated**: 50 advanced LSTM features
- **Model Performance**: HBEA 50% accuracy, GDEA 43% accuracy
- **Processing Efficiency**: 7.4 minutes for 2,617 documents with parallel processing