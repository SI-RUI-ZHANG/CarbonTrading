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
    is_open: pd.Series,
    capital: float = INIT_CAPITAL,
) -> Tuple[SignalSeries, NavSeries]:
    """
    LEGACY VERSION - kept for compatibility
    @param close: daily close prices
    @param signal: position to hold *tomorrow* (1 = long, 0 = cash), same index
    @param is_open: whether the position is a trading day (1 = open, 0 = close)
    @param capital: initial capital
    @return: tuple of signal and NAV series
    """
    cash, units = capital, 0
    nav = []


    # To avoid look-ahead bias, we shift the signal by one day.
    for p, s, open_today in zip(close, signal.shift(fill_value=0), is_open):
        if not open_today:
            # Non-trading day: don't update position, just NAV.
            nav.append(cash + units * p)
            continue

        # Trading day: update positions based on signals
        if s == 1 and units == 0:  # open long
            units, cash = divmod(cash, p)
        elif s == 0 and units > 0:  # liquidate
            cash += units * p
            units = 0
        nav.append(cash + units * p)

        prev_signal = s  # remember signal from this valid trading day

    return SignalSeries(signal, index=close.index, name="Signal"), NavSeries(
        nav, index=close.index, name="NAV"
    )


def simulate_nav_trading_only(
    close: pd.Series,
    signal: pd.Series,
    capital: float = INIT_CAPITAL,
) -> Tuple[SignalSeries, NavSeries]:
    """
    NEW VERSION for trading-only data
    @param close: trading day close prices only
    @param signal: position to hold *tomorrow* (1 = long, 0 = cash), same index
    @param capital: initial capital
    @return: tuple of signal and NAV series
    """
    cash, units = capital, 0
    nav = []

    # To avoid look-ahead bias, we shift the signal by one day.
    for p, s in zip(close, signal.shift(fill_value=0)):
        # All days are trading days now
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
    def run(self, close, is_open=None, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        # Support both old and new interfaces
        if is_open is not None:
            # Legacy mode with is_open
            first_idx = is_open[is_open].index[0]
            signal = pd.Series(0, index=close.index, name="Signal")
            signal.loc[first_idx:] = 1  # long from first open day
            _, nav = simulate_nav(close, signal, is_open, capital)
        else:
            # Trading-only mode
            signal = pd.Series(1, index=close.index, name="Signal")  # Always long
            _, nav = simulate_nav_trading_only(close, signal, capital)
        return signal, nav


class SMA20(Strategy):
    def __init__(self, window=20):
        self.window = window

    def run(self, close, is_open=None, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        sma = close.rolling(self.window).mean()
        signal = (close > sma).astype(int).rename("Signal")
        
        if is_open is not None:
            # Legacy mode
            return simulate_nav(close, signal, is_open, capital)
        else:
            # Trading-only mode - SMA is now correctly 20 TRADING days
            return simulate_nav_trading_only(close, signal, capital)


class RSI143070(Strategy):
    def __init__(self, length=14, lo=30, hi=70):
        self.length, self.lo, self.hi = length, lo, hi

    def run(self, close, is_open=None, capital=INIT_CAPITAL) -> Tuple[SignalSeries, NavSeries]:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.length, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss)

        signal = np.select([rsi < self.lo, rsi > self.hi], [1, 0], default=np.nan)
        signal = pd.Series(signal, index=close.index).ffill().fillna(0).rename("Signal")
        
        if is_open is not None:
            # Legacy mode
            return simulate_nav(close, signal, is_open, capital)
        else:
            # Trading-only mode - RSI calculated on trading days only
            return simulate_nav_trading_only(close, signal, capital)
