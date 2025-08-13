# Detailed Report Structure for Carbon Trading LSTM Project

## CLIENT SPECIFICATION (1 page, NOT counted in 3,000 words)
**Client:** Carbon Capital Partners (fictitious institutional investment firm)
**Investment Mandate:** Deploy systematic trading strategy in Chinese carbon markets (GDEA/HBEA)
**Requirements:**
- Risk-adjusted returns exceeding buy-and-hold by 15%
- Daily trading signals with weekly liquidity
- Maximum drawdown < 20%, Sharpe ratio > 0.8
- No shorting, max 10% average daily volume
- Monthly reporting with audit trail

## MAIN REPORT (3,000 words total)

### 1. EXECUTIVE SUMMARY (250 words)
- **Investment Decision:** Go/no-go recommendation for pilot allocation
- **Strategy Overview:** LSTM-driven long-only systematic trading
- **Key Metrics:** Expected return, volatility, Sharpe ratio, capacity
- **Risk Limits:** Drawdown controls, position limits, turnover caps

### 2. INTRODUCTION (250 words)
- **Market Opportunity:** Chinese carbon market inefficiencies
- **Problem Statement:** Monetizing policy-driven price dislocations
- **Scope:** GDEA/HBEA markets, daily signals, no derivatives
- **Value Proposition:** Superior risk-adjusted returns via systematic approach

### 3. DATA (400 words)
#### 3.1 Data Sources
- Carbon prices: GDEA/HBEA daily (2014-2024)
- Macroeconomic: 21 indicators with appropriate lags
- Policy documents: 3,312 scraped, 989 scored for sentiment
- Volume and liquidity metrics

#### 3.2 Processing Pipeline
- Trading calendar alignment (XSHG)
- Forward-fill → interpolation → daily alignment
- Leakage prevention: rolling normalization on past data only
- Train/validation/test split: 60/20/20 chronological

**Required Figure:** Price series with policy events annotated
**Required Table:** Data summary statistics

### 4. METHODOLOGY (600 words)
#### 4.1 Trading Strategy Design
- **Signal Type:** Classification (up/down-flat) with probability thresholds
- **Horizon:** 5-day forward returns
- **Entry Rules:** Long when P(up) > threshold, calibrated by validation
- **Position Sizing:** Volatility targeting (8-12% annualized)

#### 4.2 LSTM Architecture
- 2-layer LSTM, 64 hidden units
- 60-day lookback window
- 50 input features (macro + technical + sentiment)
- BCEWithLogitsLoss, class balancing

#### 4.3 Sentiment Integration
- Regional separation: GDEA (MEE+GZETS), HBEA (MEE+HBETS)
- Document scoring: -150 to +150 spectrum
- Exponential decay: 7-day half-life
- 12 sentiment features per market

#### 4.4 Walk-Forward Validation
- 8 walks (daily), 14 walks (weekly)
- No data leakage between periods
- Hyperparameter selection on validation only

#### 4.5 Baseline Strategies
- Buy-and-hold
- 20/60 SMA crossover
- Gradient boosting (same features)

**Required Figure:** Strategy pipeline schematic
**Required Table:** Model configuration summary

### 5. RESULTS & ANALYSIS (750 words)
#### 5.1 Out-of-Sample Performance
- **GDEA Results:** ~51% accuracy, Sharpe ratio, returns
- **HBEA Results:** ~47% accuracy, Sharpe ratio, returns
- **Sentiment Impact:** +20% F1 for GDEA weekly, mixed for HBEA

#### 5.2 Risk Metrics
- Maximum drawdown analysis
- Value at Risk (95%, 99%)
- Turnover and capacity constraints
- Stress testing under policy shocks

#### 5.3 Transaction Cost Analysis
- Three scenarios: 5/15/30 bps per side
- Slippage modeling based on participation
- Break-even analysis

#### 5.4 Meta-Model Enhancement
- Error reversal approach
- 100% coverage, no abstention
- Incremental performance improvement

**Required Figures:** 
- Equity curves (strategy vs benchmarks)
- Drawdown chart
- Sharpe ratio comparison bars

**Required Table:** Performance summary with confidence intervals

### 6. DISCUSSION & EVALUATION (400 words)
#### 6.1 Investment Interpretation
- Conditions for strategy success
- Capacity limits (CNY amount at 5% ADV)
- Operational readiness assessment

#### 6.2 Limitations
- Short sample period (limited China ETS history)
- Liquidity constraints in smaller market
- Model drift and regime change risks
- Policy shock vulnerability

#### 6.3 Risk Controls
- Stop-loss at 15% drawdown
- Volatility targeting mechanism
- Model monitoring and retraining schedule
- Fallback to baseline strategy

### 7. IMPLEMENTATION FRAMEWORK (300 words)
#### 7.1 Technical Infrastructure
- Real-time data pipeline requirements
- Model deployment architecture
- Execution system (VWAP/TWAP)

#### 7.2 Operational Workflow
- End-of-day signal generation
- Next-day execution protocol
- Performance monitoring dashboard

#### 7.3 Cost-Benefit Analysis
- Infrastructure costs
- Expected returns net of costs
- Scalability assessment

### 8. CONCLUSION (150 words)
- **Clear Recommendation:** Proceed with limited pilot (10-20% of sleeve)
- **Success Criteria:** Sharpe > 0.8, max DD < 15% over 6 months
- **Next Steps:** Deploy pilot, monitor KPIs, scale if successful

## APPENDICES (Not counted)
**A. Technical Details**
- Full model architectures and hyperparameters
- Feature importance analysis
- Cross-validation results

**B. Detailed Performance**
- Complete walk-forward results
- Statistical significance tests
- Monte Carlo simulations

**C. Code Repository**
- GitHub structure
- Key scripts documentation
- Reproducibility instructions

**D. Data Quality**
- Document corpus statistics
- Missing data handling
- Feature engineering details

## Key Implementation Notes:
1. **Avoid Look-Ahead Bias:** Use strict chronological splits, compute features on past data only
2. **Realistic Costs:** Include bid-ask spreads, slippage, market impact
3. **Conservative Capacity:** Assume max 5-10% of average daily volume
4. **Clear Narrative:** Focus on investment decision, not model complexity
5. **Professional Tone:** Write for institutional investor audience, not academic
6. **Quantitative Support:** Every claim backed by metrics from out-of-sample testing

This structure ensures compliance with all project requirements while presenting your sophisticated LSTM carbon trading system as a practical investment tool.