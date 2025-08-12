"""
Comparison visualization functions
Creates aligned plots comparing multiple models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from plot_config import *

def align_to_common_start(nav_dict: Dict[str, pd.Series]) -> Tuple[Dict[str, pd.Series], pd.Timestamp]:
    """
    Align all NAV series to start from the same date (latest common start)
    
    Args:
        nav_dict: Dictionary of NAV series
        
    Returns:
        aligned_dict: Dictionary of aligned NAV series
        start_date: Common start date
    """
    # Find the latest start date
    start_dates = [nav.index[0] for nav in nav_dict.values()]
    common_start = max(start_dates)
    
    # Align all series
    aligned_dict = {}
    for name, nav in nav_dict.items():
        aligned_nav = nav[nav.index >= common_start].copy()
        # Rebase to 100 at start
        if len(aligned_nav) > 0:
            aligned_nav = (aligned_nav / aligned_nav.iloc[0]) * 100
            aligned_dict[name] = aligned_nav
    
    return aligned_dict, common_start

def plot_cumulative_returns_comparison(nav_dict: Dict[str, pd.Series],
                                      title_suffix: str = "",
                                      align_start: bool = True,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot aligned cumulative returns comparison
    
    Args:
        nav_dict: Dictionary of NAV series
        title_suffix: Additional text for title
        align_start: Whether to align to common start date
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['comparison'])
    
    if align_start:
        # Align to common start
        aligned_dict, start_date = align_to_common_start(nav_dict)
        title_prefix = "Aligned "
        start_text = f" (from {start_date.strftime('%Y-%m-%d')})"
    else:
        aligned_dict = nav_dict
        title_prefix = ""
        start_text = ""
    
    # Plot each model
    for name, nav in aligned_dict.items():
        if align_start:
            # Already rebased to 100
            cum_returns = nav - 100
        else:
            cum_returns = (nav / nav.iloc[0] - 1) * 100
        
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = MODEL_LINESTYLES.get(name, '-')
        linewidth = MODEL_LINEWIDTHS.get(name, 2)
        alpha = MODEL_ALPHAS.get(name, 1.0)
        
        ax.plot(cum_returns.index, cum_returns.values,
               label=name, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha)
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Cumulative Return (%)', fontsize=14)
    ax.set_title(f'{title_prefix}Cumulative Returns Comparison{title_suffix}{start_text}',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    create_legend(ax, loc='best', ncol=len(aligned_dict) if len(aligned_dict) <= 4 else 2)
    
    # Add shaded regions for market events (optional)
    # ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-06-01'), 
    #           alpha=0.1, color='red', label='COVID-19')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_drawdown_comparison(nav_dict: Dict[str, pd.Series],
                            title_suffix: str = "",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot drawdown comparison with transparency
    
    Args:
        nav_dict: Dictionary of NAV series
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['comparison'])
    
    # Align to common start
    aligned_dict, start_date = align_to_common_start(nav_dict)
    
    # Calculate and plot drawdowns
    for name, nav in aligned_dict.items():
        running_max = nav.expanding().max()
        drawdown = ((nav - running_max) / running_max) * 100
        
        color = MODEL_COLORS.get(name, '#333333')
        alpha = 0.5  # Use transparency for overlapping
        
        ax.fill_between(drawdown.index, 0, drawdown.values,
                       color=color, alpha=alpha * 0.3, label=name)
        ax.plot(drawdown.index, drawdown.values,
               color=color, linewidth=1.5, alpha=alpha)
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=14)
    ax.set_title(f'Drawdown Comparison{title_suffix} (from {start_date.strftime("%Y-%m-%d")})',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    create_legend(ax, loc='lower left')
    
    # Set y-axis limits
    all_dd = []
    for nav in aligned_dict.values():
        running_max = nav.expanding().max()
        dd = ((nav - running_max) / running_max) * 100
        all_dd.append(dd.min())
    ax.set_ylim([min(all_dd) * 1.1, 5])
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_sharpe_comparison(nav_dict: Dict[str, pd.Series],
                                  window: int = 252,
                                  title_suffix: str = "",
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot rolling Sharpe ratio comparison
    
    Args:
        nav_dict: Dictionary of NAV series
        window: Rolling window size
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['comparison'])
    
    # Align to common start
    aligned_dict, start_date = align_to_common_start(nav_dict)
    
    # Calculate and plot rolling Sharpe for each model
    for name, nav in aligned_dict.items():
        returns = nav.pct_change().dropna()
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = MODEL_LINESTYLES.get(name, '-')
        linewidth = 2
        alpha = MODEL_ALPHAS.get(name, 1.0)
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
               label=name, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=-1, color='red', linestyle=':', linewidth=1, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sharpe Ratio', fontsize=14)
    ax.set_title(f'Rolling Sharpe Ratio Comparison ({window}-day){title_suffix}',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    create_legend(ax, loc='upper left')
    
    # Set reasonable y-axis limits
    ax.set_ylim([-3, 3])
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_volatility_comparison(nav_dict: Dict[str, pd.Series],
                                      window: int = 60,
                                      title_suffix: str = "",
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot rolling volatility comparison
    
    Args:
        nav_dict: Dictionary of NAV series
        window: Rolling window size
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['comparison'])
    
    # Align to common start
    aligned_dict, start_date = align_to_common_start(nav_dict)
    
    # Calculate and plot rolling volatility for each model
    for name, nav in aligned_dict.items():
        returns = nav.pct_change().dropna()
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = MODEL_LINESTYLES.get(name, '-')
        linewidth = 2
        alpha = MODEL_ALPHAS.get(name, 0.8)
        
        ax.plot(rolling_vol.index, rolling_vol.values,
               label=name, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=14)
    ax.set_title(f'Rolling Volatility Comparison ({window}-day){title_suffix}',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    create_legend(ax, loc='upper right')
    
    # Set y-axis limits
    ax.set_ylim(bottom=0)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_performance_ribbon(nav_dict: Dict[str, pd.Series],
                          metric: str = 'sharpe',
                          window: int = 252,
                          title_suffix: str = "",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance ribbon chart showing relative rankings over time
    
    Args:
        nav_dict: Dictionary of NAV series
        metric: 'sharpe', 'returns', or 'volatility'
        window: Rolling window for metrics
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('clean')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['comparison'])
    
    # Align to common start
    aligned_dict, start_date = align_to_common_start(nav_dict)
    
    # Calculate metric for each model
    metric_dict = {}
    dates = None
    
    for name, nav in aligned_dict.items():
        returns = nav.pct_change().dropna()
        
        if metric == 'sharpe':
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            values = (rolling_mean / rolling_std) * np.sqrt(252)
        elif metric == 'returns':
            values = returns.rolling(window).mean() * 252 * 100
        elif metric == 'volatility':
            values = returns.rolling(window).std() * np.sqrt(252) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_dict[name] = values
        if dates is None:
            dates = values.index
    
    # Create DataFrame and rank
    df_metrics = pd.DataFrame(metric_dict)
    
    # Rank (higher is better for sharpe/returns, lower is better for volatility)
    if metric == 'volatility':
        df_ranks = df_metrics.rank(axis=1, ascending=True)
    else:
        df_ranks = df_metrics.rank(axis=1, ascending=False)
    
    # Plot ribbons
    n_models = len(aligned_dict)
    y_positions = np.arange(1, n_models + 1)
    
    for name in df_ranks.columns:
        color = MODEL_COLORS.get(name, '#333333')
        ax.fill_between(df_ranks.index, 
                       df_ranks[name] - 0.3, 
                       df_ranks[name] + 0.3,
                       color=color, alpha=0.6, label=name)
        ax.plot(df_ranks.index, df_ranks[name],
               color=color, linewidth=2)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Rank', fontsize=14)
    ax.set_title(f'Performance Ranking Ribbon ({metric.capitalize()}){title_suffix}',
                fontsize=16, fontweight='bold')
    
    # Invert y-axis so rank 1 is at top
    ax.invert_yaxis()
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'#{i}' for i in y_positions])
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig