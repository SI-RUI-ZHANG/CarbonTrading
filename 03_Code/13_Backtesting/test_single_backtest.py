"""
Test single model backtesting to verify setup
"""

import sys
import os
sys.path.append('../')

from a_lstm_strategy import LSTMBinaryStrategy
from b_backtesting_engine import BacktestEngine, simulate_nav_simple
from c_evaluation_metrics import calculate_comprehensive_metrics
import pandas as pd
import numpy as np

# Test with daily GDEA base model
model_dir = '../../04_Models/daily_GDEA_base'
market = 'GDEA'

# Load price data
price_path = '../../02_Data_Processed/01_Carbon_Markets/01_Regional/GDEA_forward_filled.parquet'
prices_df = pd.read_parquet(price_path)
prices = prices_df['close']

print(f"Loaded {len(prices)} price points from {prices.index[0]} to {prices.index[-1]}")

# Initialize strategy
strategy = LSTMBinaryStrategy(model_dir, market, 'base')

# Get signals with dates
signals = strategy.get_signals_with_dates()
print(f"Generated {len(signals)} signals from {signals.index[0]} to {signals.index[-1]}")

# Align prices with signals
aligned_prices = prices[signals.index]
print(f"Aligned {len(aligned_prices)} prices with signals")

# Run simple backtest first
nav = simulate_nav_simple(aligned_prices, signals, 1000000)
print(f"NAV range: {nav.min():.0f} to {nav.max():.0f}")
print(f"Final NAV: {nav.iloc[-1]:.0f}")

# Calculate return
total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
print(f"Total Return: {total_return:.1f}%")

# Run with transaction costs
engine = BacktestEngine(initial_capital=1000000, transaction_cost=0.001, slippage=0.0005)
nav_with_costs, stats = engine.simulate_nav(aligned_prices, signals)
print(f"\nWith costs - Final NAV: {nav_with_costs.iloc[-1]:.0f}")
print(f"Total trades: {stats['total_trades']}")
print(f"Transaction costs: {stats['total_costs']:.0f}")

# Calculate metrics
metrics = calculate_comprehensive_metrics(nav_with_costs)
print(f"\nKey Metrics:")
print(f"  CAGR: {metrics['cagr']*100:.1f}%")
print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")