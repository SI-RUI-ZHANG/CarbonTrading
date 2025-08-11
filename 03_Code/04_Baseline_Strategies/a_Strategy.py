from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple
from typing import Dict, TypeAlias
import pandas as pd

NavSeries: TypeAlias = pd.Series
SignalSeries: TypeAlias = pd.Series
NavDict: TypeAlias = Dict[str, NavSeries]
SignalDict: TypeAlias = Dict[str, SignalSeries]

INIT_CAPITAL = 1000000


# ==================================================================================
# Strategy Base Class
# ==================================================================================
class Strategy(ABC):
    @abstractmethod
    def run(
        self, close: pd.Series, capital: float = INIT_CAPITAL
    ) -> Tuple[SignalSeries, NavSeries]:
        """
        @param close: daily close prices
        @param capital: initial capital
        @return: tuple of signal and NAV series
        """
        pass


# ==================================================================================
# Translate Signal to NAV
# ==================================================================================
def simulate_nav(
    close: pd.Series,
    signal: pd.Series,
    capital: float = INIT_CAPITAL,
) -> Tuple[SignalSeries, NavSeries]:
    """
    Simulate portfolio NAV based on trading signals (trading-only data)
    @param close: trading day close prices
    @param signal: position to hold *tomorrow* (1 = long, 0 = cash), same index
    @param capital: initial capital
    @return: tuple of signal and NAV series
    """
    cash, units = capital, 0
    nav = []

    # To avoid look-ahead bias, we shift the signal by one day.
    for p, s in zip(close, signal.shift(fill_value=0)):
        # All days are trading days in our cleaned data
        if s == 1 and units == 0:  # open long
            units, cash = divmod(cash, p)
        elif s == 0 and units > 0:  # liquidate
            cash += units * p
            units = 0
        nav.append(cash + units * p)

    return SignalSeries(signal, index=close.index, name="Signal"), NavSeries(
        nav, index=close.index, name="NAV"
    )


# ==================================================================================
# Strategy Implementations
# ==================================================================================


class BuyAndHold(Strategy):
    def run(self, close, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        # Always long strategy for trading-only data
        signal = pd.Series(1, index=close.index, name="Signal")
        _, nav = simulate_nav(close, signal, capital)
        return signal, nav


class SMA20(Strategy):
    def __init__(self, window=20):
        self.window = window

    def run(self, close, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        # SMA now correctly uses 20 TRADING days
        sma = close.rolling(self.window).mean()
        signal = (close > sma).astype(int).rename("Signal")
        return simulate_nav(close, signal, capital)


class RSI143070(Strategy):
    def __init__(self, length=14, lo=30, hi=70):
        self.length, self.lo, self.hi = length, lo, hi

    def run(self, close, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        # RSI calculated on trading days only
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.length, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        signal = np.select([rsi < self.lo, rsi > self.hi], [1, 0], default=np.nan)
        signal = pd.Series(signal, index=close.index).ffill().fillna(0).rename("Signal")
        return simulate_nav(close, signal, capital)
