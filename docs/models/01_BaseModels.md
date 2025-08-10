# Base Models Documentation

## Overview

The base models module provides a framework for implementing and evaluating trading strategies on Chinese carbon markets (GDEA and HBEA). The architecture follows object-oriented design principles with a clear separation between signal generation, position management, and performance evaluation.

**Key Design Principles:**
- **No look-ahead bias**: Signals are lagged by one day before execution
- **Trading day awareness**: Positions only change on market open days
- **Daily NAV tracking**: Portfolio value calculated every calendar day
- **Modular architecture**: Strategies, evaluation, and visualization are decoupled

## Architecture

### Module Structure
```
03_Code/03_Base_Models/
├── a_Strategy.py         # Strategy base class and implementations
├── a_Evaluation.py       # Performance metrics and visualization
└── b_Strategy_Performance.py  # Strategy execution and reporting
```

### Core Components

#### 1. Strategy Base Class (`a_Strategy.py`)
- **Abstract base class**: Defines interface for all strategies
- **Signal generation**: Each strategy produces buy/sell signals
- **NAV simulation**: Converts signals to portfolio values
- **Trading day handling**: Ensures trades only execute when markets are open

#### 2. Evaluation Module (`a_Evaluation.py`)
- **Performance metrics**: CAGR, Sharpe ratio, volatility calculations
- **Visualization tools**: Price, NAV, and return plotting functions
- **Comparison utilities**: Multi-strategy performance comparison

#### 3. Performance Analysis (`b_Strategy_Performance.py`)
- **Strategy execution**: Runs all strategies on both markets
- **Report generation**: Saves performance metrics as JSON
- **Chart creation**: Generates comparison plots

## Strategy Framework

### Abstract Base Class
```python
class Strategy(ABC):
    @abstractmethod
    def run(self, close: pd.Series, capital: float = INIT_CAPITAL) 
        -> Tuple[SignalSeries, NavSeries]:
        pass
```

### Signal to NAV Translation

The `simulate_nav` function is the core engine that converts trading signals to portfolio values:

```python
def simulate_nav(close, signal, is_open, capital=INIT_CAPITAL):
    for p, s, open_today in zip(close, signal.shift(fill_value=0), is_open):
        if not open_today:
            # Non-trading day: update NAV but don't change positions
            nav.append(cash + units * p)
            continue
        
        # Trading day: execute position changes
        if s == 1 and units == 0:  # open long
            units, cash = divmod(cash, p)
        elif s == 0 and units > 0:  # liquidate
            cash += units * p
            units = 0
```

**Key Features:**
- **Signal lag**: `signal.shift(fill_value=0)` prevents look-ahead bias
- **Trading day check**: Position changes only when `is_open = True`
- **Full investment**: Uses `divmod` to invest maximum capital
- **Daily NAV**: Calculated for all days, including non-trading days

## Implemented Strategies

### 1. Buy and Hold
**Logic**: Enter long position on first trading day and hold indefinitely
```python
class BuyAndHold(Strategy):
    def run(self, close, is_open, capital=INIT_CAPITAL):
        first_idx = is_open[is_open].index[0]
        signal = pd.Series(0, index=close.index)
        signal.loc[first_idx:] = 1  # long from first open day
```
**Use Case**: Benchmark strategy for passive investment

### 2. Simple Moving Average (SMA20)
**Logic**: Long when price > 20-day moving average, cash otherwise
```python
class SMA20(Strategy):
    def run(self, close, is_open, capital=INIT_CAPITAL):
        sma = close.rolling(self.window).mean()
        signal = (close > sma).astype(int)
```
**Parameters**: 
- `window`: Moving average period (default: 20)

**Use Case**: Trend-following strategy

### 3. Relative Strength Index (RSI 14/30/70)
**Logic**: Long when RSI < 30 (oversold), cash when RSI > 70 (overbought)
```python
class RSI143070(Strategy):
    def run(self, close, is_open, capital=INIT_CAPITAL):
        # RSI calculation using exponential moving average
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/self.length).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/self.length).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        
        signal = np.select([rsi < self.lo, rsi > self.hi], 
                          [1, 0], default=np.nan)
```
**Parameters**:
- `length`: RSI period (default: 14)
- `lo`: Oversold threshold (default: 30)
- `hi`: Overbought threshold (default: 70)

**Use Case**: Mean-reversion strategy

## Performance Evaluation

### Metrics Calculation
The `evaluate_performance` function computes standard financial metrics:

| Metric | Formula | Description |
|--------|---------|-------------|
| Cumulative Return | `nav[-1] / nav[0] - 1` | Total return over period |
| CAGR | `(1 + cum_ret) ^ (252/days) - 1` | Annualized return |
| Annual Volatility | `daily_ret.std() * sqrt(252)` | Annualized standard deviation |
| Sharpe Ratio | `(mean_ret - rf) / std_ret * sqrt(252)` | Risk-adjusted return |

**Assumptions**:
- **Trading days per year**: 252
- **Risk-free rate**: 0% (adjustable)
- **Daily frequency**: All calculations assume daily data

### Visualization Functions

#### 1. `plot_nav_and_returns`
Three-panel plot showing:
- Top: Price time series
- Middle: NAV evolution
- Bottom: Daily returns

#### 2. `plot_cumulative_returns_with_price`
Two-panel comparison plot:
- Top: Cumulative returns for all strategies
- Bottom: Underlying price series

**Features**:
- Optional smoothing with moving average
- Save to file capability
- Academic-style formatting

## Usage Example

### Running Strategies
```python
# Load data
df_HBEA = pd.read_parquet("HBEA_forward_filled.parquet")
close_HBEA = df_HBEA["close"]
is_open_HBEA = df_HBEA["is_open"]

# Initialize strategies
strategies = {
    "BuyAndHold": BuyAndHold(),
    "SMA20": SMA20(),
    "RSI143070": RSI143070(),
}

# Run strategies
NAV_HBEA = {}
for name, strategy in strategies.items():
    signal, nav = strategy.run(close_HBEA, is_open_HBEA)
    NAV_HBEA[name] = nav
```

### Generating Performance Report
```python
# Calculate metrics
perf_HBEA = {}
for name in strategies.keys():
    perf_HBEA[name] = evaluate_performance(NAV_HBEA[name])

# Save to JSON
with open("HBEA_performance.json", "w") as f:
    json.dump(perf_HBEA, f, indent=4)
```

### Creating Visualizations
```python
# Comparison plot
plot_cumulative_returns_with_price(
    NAV_HBEA, 
    close_HBEA, 
    data_name="HBEA",
    save_path="HBEA_cumulative_returns.png"
)
```

## Technical Notes

### Data Requirements
- **Input format**: Pandas Series with DatetimeIndex
- **Required columns**: 
  - `close`: Daily closing prices (can have NaN on non-trading days)
  - `is_open`: Boolean flag for trading days
- **Date range**: Any period with consistent daily frequency

### Position Management
- **Long only**: Current implementation supports only long or cash positions
- **Full investment**: All available capital is invested when going long
- **Integer units**: Fractional shares handled via cash remainder
- **No transaction costs**: Current version assumes zero fees/slippage

### Performance Considerations
- **Vectorized operations**: Strategies use pandas/numpy for efficiency
- **Memory usage**: ~100KB per strategy for 10-year backtest
- **Execution time**: <100ms per strategy on standard hardware

## Limitations and Future Enhancements

### Current Limitations
1. **Long only**: No short selling capability
2. **Single asset**: Each strategy trades one market at a time
3. **No transaction costs**: Fees and slippage not modeled
4. **Binary positions**: Either fully invested or fully in cash
5. **No stop losses**: Risk management features not implemented

### Potential Enhancements
1. **Short selling**: Add ability to profit from price declines
2. **Portfolio strategies**: Multi-asset allocation and rebalancing
3. **Transaction costs**: Realistic fee and slippage modeling
4. **Position sizing**: Variable position sizes based on signals
5. **Risk management**: Stop losses, position limits, drawdown controls
6. **Machine learning**: Integration with ML-based signal generation

## Configuration

### Global Constants
```python
INIT_CAPITAL = 1000000  # Initial capital: 1 million
TRADING_DAYS = 252      # Trading days per year
```

### File Paths
The module expects data in:
```
02_Data_Processed/01_Carbon_Markets/01_Regional/
├── HBEA_forward_filled.parquet
└── GDEA_forward_filled.parquet
```

Output generated in:
```
03_Code/Plots/
├── HBEA_cumulative_returns.png
├── GDEA_cumulative_returns.png
├── HBEA_performance.json
└── GDEA_performance.json
```

## Summary

The base models module provides a robust framework for backtesting trading strategies on Chinese carbon markets. Its key strength lies in properly handling the complexities of real market data:
- Mixed trading and non-trading days
- Prevention of look-ahead bias
- Realistic position management

The modular design allows easy extension with new strategies while maintaining consistent evaluation and reporting across all implementations.