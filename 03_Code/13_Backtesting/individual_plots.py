"""
Individual plot generation functions with improved styling
Creates various single-metric visualizations with publication-quality output
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from scipy import stats
from plot_config import *

def ensure_dir_exists(filepath: str):
    """Ensure directory exists for the given filepath"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

def plot_cumulative_returns(nav_dict: Dict[str, pd.Series],
                           title_suffix: str = "",
                           benchmark_nav: Optional[pd.Series] = None,
                           walk_periods: Optional[List[Tuple]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced cumulative returns plot with walk-forward period shading
    
    Args:
        nav_dict: Dictionary of NAV series for each model
        title_suffix: Additional text for title
        benchmark_nav: Optional benchmark NAV for comparison
        walk_periods: List of (start, end) tuples for walk-forward periods
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Add walk-forward period shading if provided
    if walk_periods:
        for i, (start, end) in enumerate(walk_periods):
            if i == 0:
                label = 'Walk-Forward Periods'
            else:
                label = ''
            color = 'lightblue' if i % 2 == 0 else 'lightgray'
            ax.axvspan(start, end, alpha=0.1, color=color, label=label)
    
    # Plot each model
    for name, nav in nav_dict.items():
        cum_returns = (nav / nav.iloc[0] - 1) * 100
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = MODEL_LINESTYLES.get(name, '-')
        alpha = MODEL_ALPHAS.get(name, 1.0)
        
        ax.plot(cum_returns.index, cum_returns.values,
               color=color, linestyle=linestyle, linewidth=2.5,
               alpha=alpha, label=name)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cumulative Returns{title_suffix}', fontsize=16, fontweight='bold')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Enhanced legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                      shadow=True, framealpha=0.95)
    
    # Add final values annotation
    y_offset = 0
    for name, nav in nav_dict.items():
        cum_returns = (nav / nav.iloc[0] - 1) * 100
        final_return = cum_returns.iloc[-1]
        color = MODEL_COLORS.get(name, '#333333')
        
        ax.annotate(f'{name}: {final_return:.1f}%',
                   xy=(cum_returns.index[-1], final_return),
                   xytext=(10, y_offset), textcoords='offset points',
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=color, alpha=0.8))
        y_offset += 20
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_drawdown(nav: pd.Series,
                 title_suffix: str = "",
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced drawdown plot with duration analysis
    
    Args:
        nav: NAV series
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Calculate drawdown
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    
    # Plot drawdown with duration analysis
    ax.fill_between(drawdown.index, 0, drawdown.values * 100,
                    color='red', alpha=0.3, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values * 100,
           color='darkred', linewidth=1.5)
    
    # Mark maximum drawdown point
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min() * 100
    ax.scatter(max_dd_idx, max_dd_val, color='red', s=100, zorder=5)
    ax.annotate(f'Max: {max_dd_val:.1f}%\n{max_dd_idx.strftime("%Y-%m-%d")}',
               xy=(max_dd_idx, max_dd_val),
               xytext=(10, -10), textcoords='offset points',
               fontsize=9, color='darkred',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Calculate drawdown duration
    dd_start = None
    max_duration = 0
    current_duration = 0
    
    for i, val in enumerate(drawdown.values):
        if val < 0:
            if dd_start is None:
                dd_start = i
            current_duration = i - dd_start + 1
            max_duration = max(max_duration, current_duration)
        else:
            dd_start = None
            current_duration = 0
    
    # Add duration info to title
    days_text = f" (Max Duration: {max_duration} days)" if max_duration > 0 else ""
    
    # Add recovery periods shading
    recovery_start = None
    for i in range(1, len(drawdown)):
        if drawdown.iloc[i-1] < 0 and drawdown.iloc[i] >= -0.001:
            if recovery_start is not None:
                ax.axvspan(recovery_start, drawdown.index[i], 
                          alpha=0.1, color='green')
            recovery_start = None
        elif drawdown.iloc[i] < -0.001 and recovery_start is None:
            recovery_start = drawdown.index[i]
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-10, color='orange', linestyle='--', linewidth=0.5, alpha=0.5, label='10% DD')
    ax.axhline(y=-20, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='20% DD')
    
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Drawdown Analysis{title_suffix}{days_text}', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Add legend
    ax.legend(loc='lower left', framealpha=0.9)
    
    # Set y-axis limits
    ax.set_ylim([drawdown.min() * 100 * 1.2, 5])
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_sharpe(nav: pd.Series,
                       title_suffix: str = "",
                       window: int = 252,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced rolling Sharpe ratio plot with smoothed trend
    
    Args:
        nav: NAV series
        title_suffix: Additional text for title
        window: Rolling window size
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Calculate returns and rolling Sharpe
    returns = nav.pct_change().dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    # Plot rolling Sharpe with smoothed trend
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
           color='steelblue', linewidth=1.5, alpha=0.6, label=f'{window}-day Rolling')
    
    # Add smoothed trend line
    if len(rolling_sharpe.dropna()) > 20:
        smooth_trend = add_smooth_trend(ax, rolling_sharpe.dropna(), 
                                       color='darkblue', label='Smoothed Trend')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='SR = 1.0')
    ax.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='SR = -1.0')
    ax.axhline(y=0.5, color='yellow', linestyle=':', linewidth=1, alpha=0.5, label='SR = 0.5')
    
    # Fill areas
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                   where=(rolling_sharpe >= 0), color='green', alpha=0.1)
    ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                   where=(rolling_sharpe < 0), color='red', alpha=0.1)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.set_title(f'Rolling Sharpe Ratio ({window}-day){title_suffix}', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter')
    
    # Enhanced legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add current value
    current_sharpe = rolling_sharpe.iloc[-1]
    if not np.isnan(current_sharpe):
        color = 'green' if current_sharpe > 0 else 'red'
        ax.annotate(f'Current: {current_sharpe:.2f}',
                   xy=(rolling_sharpe.index[-1], current_sharpe),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=color, alpha=0.8))
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_monthly_heatmap(nav: pd.Series,
                        title_suffix: str = "",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced monthly returns heatmap centered at zero with yearly totals
    
    Args:
        nav: NAV series
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('clean')
    
    # Calculate monthly returns
    monthly_nav = nav.resample('ME').last()
    monthly_returns = monthly_nav.pct_change().dropna() * 100
    
    # Create matrix
    years = sorted(monthly_returns.index.year.unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Include yearly totals column
    returns_matrix = pd.DataFrame(index=years, columns=months + ['Year'])
    
    for date, ret in monthly_returns.items():
        returns_matrix.loc[date.year, months[date.month - 1]] = ret
    
    # Calculate yearly returns
    for year in years:
        year_data = returns_matrix.loc[year, months].dropna()
        if len(year_data) > 0:
            # Compound monthly returns properly
            year_return = ((1 + year_data / 100).prod() - 1) * 100
            returns_matrix.loc[year, 'Year'] = year_return
    
    # Separate data for plotting
    plot_matrix = returns_matrix[months]
    year_totals = returns_matrix[['Year']]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                   gridspec_kw={'width_ratios': [12, 1]})
    
    # Determine color scale centered at zero
    vmax = max(abs(plot_matrix.min().min()), abs(plot_matrix.max().max()))
    vmin = -vmax
    
    # Main heatmap
    sns.heatmap(plot_matrix.astype(float), 
               annot=True, fmt='.1f', 
               cmap='RdYlGn', center=0,
               vmin=vmin, vmax=vmax,
               cbar_kws={'label': 'Monthly Return (%)'},
               linewidths=0.5, linecolor='gray',
               ax=ax1)
    
    # Yearly totals heatmap
    sns.heatmap(year_totals.astype(float),
               annot=True, fmt='.1f',
               cmap='RdYlGn', center=0,
               vmin=vmin, vmax=vmax,
               cbar=False,
               linewidths=0.5, linecolor='gray',
               ax=ax2)
    
    # Formatting
    ax1.set_title(f'Monthly Returns Heatmap{title_suffix}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Year', fontsize=14, fontweight='bold')
    ax2.set_title('Yearly', fontsize=14, fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    
    # Rotate labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_return_distribution(returns: pd.Series,
                            title_suffix: str = "",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced return distribution with KDE overlay and statistical tests
    
    Args:
        returns: Returns series (already in percentage)
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Histogram with KDE and normal overlay
    n, bins, patches = ax1.hist(returns, bins=50, density=True,
                                alpha=0.6, color='steelblue', 
                                edgecolor='black', linewidth=0.5)
    
    # Add KDE
    kde = stats.gaussian_kde(returns)
    x_range = np.linspace(returns.min(), returns.max(), 200)
    ax1.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    
    # Fit and plot normal distribution
    mu, std = returns.mean(), returns.std()
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    p_norm = stats.norm.pdf(x_norm, mu, std)
    ax1.plot(x_norm, p_norm, 'r--', linewidth=2, label='Normal fit')
    
    # Add vertical lines for statistics
    ax1.axvline(mu, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mu:.2f}%')
    ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Median: {returns.median():.2f}%')
    
    # VaR and CVaR lines
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    ax1.axvline(var_95, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, 
               label=f'VaR(95%): {var_95:.2f}%')
    ax1.axvline(cvar_95, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'CVaR(95%): {cvar_95:.2f}%')
    
    ax1.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Enhanced Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('steelblue')
    ax2.get_lines()[0].set_markeredgecolor('black')
    ax2.get_lines()[0].set_markersize(5)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2)
    
    ax2.set_title('Q-Q Plot vs Normal', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics box with normality tests
    jarque_bera = stats.jarque_bera(returns)
    shapiro = stats.shapiro(returns) if len(returns) <= 5000 else (np.nan, np.nan)
    
    textstr = f'Statistics:\n'
    textstr += f'Mean: {mu:.3f}%\n'
    textstr += f'Std: {std:.3f}%\n'
    textstr += f'Skew: {returns.skew():.3f}\n'
    textstr += f'Kurt: {returns.kurtosis():.3f}\n'
    textstr += f'JB p-val: {jarque_bera[1]:.4f}\n'
    if not np.isnan(shapiro[1]):
        textstr += f'SW p-val: {shapiro[1]:.4f}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    fig.suptitle(f'Return Distribution Analysis{title_suffix}', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_metrics(nav: pd.Series,
                        metrics: List[str] = ['volatility', 'sharpe'],
                        window: int = 60,
                        title_suffix: str = "",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple rolling metrics in subplots
    
    Args:
        nav: NAV series
        metrics: List of metrics to plot
        window: Rolling window size
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    # Calculate returns
    returns = nav.pct_change().dropna()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric == 'volatility':
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
            ax.plot(rolling_vol.index, rolling_vol.values,
                   color='steelblue', linewidth=1.5)
            ax.fill_between(rolling_vol.index, 0, rolling_vol.values,
                          alpha=0.3, color='steelblue')
            ax.set_ylabel('Volatility (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{window}-day Rolling Volatility', fontsize=14, fontweight='bold')
            
        elif metric == 'sharpe':
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
            
            ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                   color='green', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                          where=(rolling_sharpe >= 0), color='green', alpha=0.2)
            ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                          where=(rolling_sharpe < 0), color='red', alpha=0.2)
            ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
            ax.set_title(f'{window}-day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        format_axis_dates(ax, 'quarter')
    
    axes[-1].set_xlabel('Date', fontsize=14, fontweight='bold')
    
    fig.suptitle(f'Rolling Metrics Analysis{title_suffix}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig