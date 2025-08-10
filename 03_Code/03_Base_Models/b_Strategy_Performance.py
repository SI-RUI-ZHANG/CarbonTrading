import pandas as pd
import numpy as np
import os
import json
from a_Strategy import BuyAndHold, SMA20, RSI143070, NavDict, SignalDict
from a_Evaluation import evaluate_performance, plot_cumulative_returns_with_price
# support for parquet
import pyarrow

base_dir = os.path.dirname(os.path.abspath(__file__))
path_HBEA = os.path.join(base_dir, "../../02_Data_Processed/01_Carbon_Markets/01_Regional/HBEA_forward_filled.parquet")
path_GDEA = os.path.join(base_dir, "../../02_Data_Processed/01_Carbon_Markets/01_Regional/GDEA_forward_filled.parquet")

df_HBEA = pd.read_parquet(path_HBEA)
df_GDEA = pd.read_parquet(path_GDEA)

# Only keep the close as pd.Series (trading-only data)
close_HBEA = df_HBEA["close"]
close_GDEA = df_GDEA["close"]

# Strategies (no is_open needed with trading-only data)
strategies = {
    "BuyAndHold": BuyAndHold(),
    "SMA20": SMA20(),
    "RSI143070": RSI143070(),
}

NAV_HBEA: NavDict = {}
SIGNAL_HBEA: SignalDict = {}
NAV_GDEA: NavDict = {}
SIGNAL_GDEA: SignalDict = {}
for name, strategy in strategies.items():
    signal, nav = strategy.run(close_HBEA)
    NAV_HBEA[name] = nav
    SIGNAL_HBEA[name] = signal
    signal, nav = strategy.run(close_GDEA)
    NAV_GDEA[name] = nav
    SIGNAL_GDEA[name] = signal

# Plot
# Create plots directory if it doesn't exist
base_dir = os.path.dirname(os.path.abspath(__file__))
path_plots = os.path.join(base_dir, "..", "Plots")

os.makedirs(path_plots, exist_ok=True)
plot_cumulative_returns_with_price(NAV_HBEA, close_HBEA, data_name="HBEA", smooth=1, save=True, save_path=os.path.join(path_plots, "HBEA_cumulative_returns.png"))
plot_cumulative_returns_with_price(NAV_GDEA, close_GDEA, data_name="GDEA", smooth=1, save=True, save_path=os.path.join(path_plots, "GDEA_cumulative_returns.png"))

# Record the performance of the strategies using a_Evaluation.evaluate_performance
# Create a dictionary to store performance metrics
perf_HBEA = {}
perf_GDEA = {}

# Calculate performance metrics for each strategy
for name in strategies.keys():
    perf_HBEA[name] = evaluate_performance(NAV_HBEA[name])
    perf_GDEA[name] = evaluate_performance(NAV_GDEA[name])

# Save performance metrics to CSV
with open(os.path.join(path_plots, "HBEA_performance.json"), "w") as f:
    json.dump(perf_HBEA, f, indent=4)
with open(os.path.join(path_plots, "GDEA_performance.json"), "w") as f:
    json.dump(perf_GDEA, f, indent=4)