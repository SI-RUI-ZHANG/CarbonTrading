"""
Debug backtesting issue
"""

import sys
import os
sys.path.append('../')

from a_lstm_strategy import LSTMBinaryStrategy
import pandas as pd
import numpy as np

# Test with daily GDEA base model
model_dir = '../../04_Models/daily_GDEA_base'
market = 'GDEA'

# Load price data
price_path = '../../02_Data_Processed/01_Carbon_Markets/01_Regional/GDEA_forward_filled.parquet'
prices_df = pd.read_parquet(price_path)
prices = prices_df['close']

# Initialize strategy
strategy = LSTMBinaryStrategy(model_dir, market, 'base')

# Get signals with dates
signals = strategy.get_signals_with_dates()

print(f"Signal statistics:")
print(f"  Total signals: {len(signals)}")
print(f"  Long signals (1): {(signals == 1).sum()}")
print(f"  Cash signals (0): {(signals == 0).sum()}")
print(f"  Signal values: {signals.unique()}")
print(f"  First 10 signals: {signals.head(10).values}")

# Check alignment
aligned_prices = prices[signals.index]
print(f"\nPrice/Signal alignment:")
print(f"  Aligned prices: {len(aligned_prices)}")
print(f"  First date: {aligned_prices.index[0]}")
print(f"  Last date: {aligned_prices.index[-1]}")

# Simple manual simulation
cash = 1000000
units = 0
nav_list = []
trades = 0

signals_shifted = signals.shift(1).fillna(0)

for i, (date, price) in enumerate(aligned_prices.items()):
    signal = signals_shifted.iloc[i]
    
    if i < 5:
        print(f"Date {date}: Price={price:.2f}, Signal={signal}, Cash={cash:.0f}, Units={units:.2f}")
    
    if signal == 1 and units == 0:  # Buy
        units = cash / price
        cash = 0
        trades += 1
        if i < 5:
            print(f"  -> BUY {units:.2f} units")
    elif signal == 0 and units > 0:  # Sell
        cash = units * price
        units = 0
        trades += 1
        if i < 5:
            print(f"  -> SELL for {cash:.0f}")
    
    nav_list.append(cash + units * price)

print(f"\nManual simulation:")
print(f"  Total trades: {trades}")
print(f"  Final NAV: {nav_list[-1]:.0f}")
print(f"  Return: {(nav_list[-1]/1000000 - 1)*100:.1f}%")