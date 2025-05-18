import pandas as pd
import numpy as np
import os

from a_Strategy import *
from a_Evaluation import *


# support for parquet
import pyarrow

base_dir = os.path.dirname(os.path.abspath(__file__))
path_HBEA = os.path.join(base_dir, "../../02_Data_Processed/01_Carbon_Markets/01_Regional/HBEA_forward_filled.parquet")
path_GDEA = os.path.join(base_dir, "../../02_Data_Processed/01_Carbon_Markets/01_Regional/GDEA_forward_filled.parquet")

df_HBEA = pd.read_parquet(path_HBEA)
df_GDEA = pd.read_parquet(path_GDEA)

# Only keep the close as pd.Series
close_HBEA = df_HBEA["close"]
is_open_HBEA = df_HBEA["is_open"]
close_GDEA = df_GDEA["close"]
is_open_GDEA = df_GDEA["is_open"]

# Strategies
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
    signal, nav = strategy.run(close_HBEA, is_open_HBEA)
    NAV_HBEA[name] = nav
    SIGNAL_HBEA[name] = signal
    signal, nav = strategy.run(close_GDEA, is_open_GDEA)
    NAV_GDEA[name] = nav
    SIGNAL_GDEA[name] = signal

# Plot
# Create plots directory if it doesn't exist
base_dir = os.path.dirname(os.path.abspath(__file__))
path_plots = os.path.join(base_dir, "..", "Plots")

os.makedirs(path_plots, exist_ok=True)
plot_cumulative_returns_with_price(NAV_HBEA, close_HBEA, data_name="HBEA", smooth=1, save=True, save_path="plots/HBEA_cumulative_returns.png")
plot_cumulative_returns_with_price(NAV_GDEA, close_GDEA, data_name="GDEA", smooth=1, save=True, save_path="plots/GDEA_cumulative_returns.png")