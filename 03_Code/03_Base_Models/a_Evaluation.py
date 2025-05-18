import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from a_Strategy import NavDict

TRADING_DAYS = 252

def evaluate_performance(
    nav: pd.Series, rf: float = 0.0, trading_days: int = TRADING_DAYS
) -> dict:
    """
    @param nav: daily NAV series
    @param rf: risk-free rate
    @param trading_days: number of trading days in a year
    @return: dictionary of performance metrics
    @note:
        - Assumes nav.index is monotonic and daily; gaps are OK.
        - Compute CumRet, CAGR (Annual Return), AnnVol, Sharpe for a daily NAV series.
    """
    if not isinstance(nav.index, pd.DatetimeIndex):
        raise TypeError("`nav` index must be a pandas.DatetimeIndex")
    nav = nav.dropna()

    daily_ret = nav.pct_change().dropna()
    # metrics
    cum_ret = nav.iloc[-1] / nav.iloc[0] - 1
    days = (nav.index[-1] - nav.index[0]).days
    cagr = (1 + cum_ret) ** (trading_days / max(days, 1)) - 1
    ann_vol = daily_ret.std(ddof=0) * np.sqrt(trading_days)
    sharpe = ((daily_ret.mean() - rf / trading_days) / daily_ret.std(ddof=0)) * np.sqrt(
        trading_days
    )

    return {
        "CumReturn": cum_ret,
        "AnnualReturn": cagr,
        "AnnualVol": ann_vol,
        "SharpeRatio": sharpe,
    }


def plot_nav_and_returns(
    close: pd.Series,
    nav: pd.Series,
    title: str = "Strategy Performance",
    smooth: int = 1,
) -> None:
    """
    Plots price, NAV, and daily returns in a clear, academic-style figure.

    @param close: daily close prices
    @param nav: daily NAV
    @param title: title of the plot
    @param smooth: rolling window for smoothing all plots (visual only)
    @return: None
    """

    # Compute daily returns
    daily_ret = nav.pct_change()

    # Optional smoothing
    close_plot = close.rolling(smooth).mean() if smooth > 1 else close
    nav_plot = nav.rolling(smooth).mean() if smooth > 1 else nav
    ret_plot = daily_ret.rolling(smooth).mean() if smooth > 1 else daily_ret

    # Set up figure
    fig, (ax_price, ax_nav, ax_ret) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]}
    )

    # Top panel: Price
    ax_price.plot(close_plot, color="black", lw=1.2, label="Price")
    ax_price.set_ylabel("Price")
    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="upper left")

    # Middle panel: NAV
    ax_nav.plot(nav_plot, color="navy", lw=1.2, label="NAV")
    ax_nav.set_ylabel("NAV")
    ax_nav.grid(alpha=0.3)
    ax_nav.legend(loc="upper left")

    # Bottom panel: Daily Returns
    ax_ret.plot(ret_plot, color="darkgreen", lw=0.8, label="Daily Return")
    ax_ret.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax_ret.set_ylabel("Return")
    ax_ret.set_xlabel("Date")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_cumulative_returns_with_price(
    navs: NavDict, 
    close: pd.Series, 
    data_name: str = "Strategy", 
    smooth: int = 1, 
    save: bool = False,
    save_path: str = ""
) -> None:
    """
    Plots cumulative returns of different strategies and price in two subplots.

    @param navs: dictionary of strategy names and NAV series
    @param close: daily close price series
    @param data_name: descriptive name for dataset/strategies
    @param smooth: rolling window for smoothing all plots (visual only)
    """

    # Prepare figure with two subplots
    fig, (ax_cumret, ax_price) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Title logic
    title = (
        f"{data_name} Comparison (Cumulative Returns)"
        if smooth == 1
        else f"{data_name} Comparison (Cumulative Returns, {smooth}-day SMA)"
    )

    # --- Top: Cumulative Returns ---
    for name, nav in navs.items():
        cum_returns = nav / nav.iloc[0] - 1
        cum_returns_smoothed = (
            cum_returns.rolling(smooth).mean() if smooth > 1 else cum_returns
        )
        ax_cumret.plot(cum_returns_smoothed, label=name, lw=1.5)

    ax_cumret.axhline(0, color="grey", linestyle="--", lw=0.8)
    ax_cumret.set_ylabel("Cumulative Return")
    ax_cumret.set_title(title, fontsize=14, fontweight="bold")
    ax_cumret.grid(alpha=0.3)
    ax_cumret.legend(loc="best")

    # --- Bottom: Price plot ---
    close_smoothed = close.rolling(smooth).mean() if smooth > 1 else close
    ax_price.plot(close_smoothed, color="black", lw=1.2, label="Price")
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("Date")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="best")

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


