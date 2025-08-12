"""
Improved individual model visualization functions
Creates separate plots for each model's performance metrics with enhanced features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from scipy import stats
from plot_config import *

def plot_cumulative_returns(nav: pd.Series, 
                           model_name: str,
                           benchmark_nav: Optional[pd.Series] = None,
                           walk_periods: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced cumulative returns plot with period shading and better annotations
    
    Args:
        nav: NAV series
        model_name: Name of the model
        benchmark_nav: Optional benchmark NAV for comparison
        walk_periods: List of (start, end) tuples for walk-forward periods
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Calculate cumulative returns
    cum_returns = (nav / nav.iloc[0] - 1) * 100
    
    # Add walk-forward period shading if provided
    if walk_periods:
        for i, (start, end) in enumerate(walk_periods):
            if start >= cum_returns.index[0] and end <= cum_returns.index[-1]:
                color = 'lightblue' if i % 2 == 0 else 'lightgray'
                ax.axvspan(start, end, alpha=0.1, color=color, zorder=0)
    
    # Plot main model
    color = MODEL_COLORS.get(model_name, '#333333')
    ax.plot(cum_returns.index, cum_returns.values, 
            color=color, linewidth=2.5, label=model_name, zorder=3)
    
    # Fill area under curve
    ax.fill_between(cum_returns.index, 0, cum_returns.values, 
                    alpha=0.2, color=color, zorder=1)
    
    # Add benchmark if provided
    if benchmark_nav is not None:
        bench_returns = (benchmark_nav / benchmark_nav.iloc[0] - 1) * 100
        ax.plot(bench_returns.index, bench_returns.values,
               color=MODEL_COLORS['Benchmark'], 
               linestyle='--', linewidth=2, 
               alpha=0.7, label='Buy & Hold', zorder=2)
    
    # Mark major drawdowns
    running_max = nav.expanding().max()
    drawdown = ((nav - running_max) / running_max) * 100
    major_dd_mask = drawdown < -20  # Mark periods with >20% drawdown
    if major_dd_mask.any():
        dd_periods = []
        in_dd = False
        start_dd = None
        for date, is_dd in major_dd_mask.items():
            if is_dd and not in_dd:
                start_dd = date
                in_dd = True
            elif not is_dd and in_dd:
                dd_periods.append((start_dd, date))
                in_dd = False
        
        for start, end in dd_periods[:3]:  # Show top 3 drawdown periods
            ax.axvspan(start, end, alpha=0.1, color='red', zorder=0)
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Cumulative Return (%)', fontsize=14)
    
    # Enhanced title with date range
    date_range = f"{cum_returns.index[0].strftime('%Y-%m')} to {cum_returns.index[-1].strftime('%Y-%m')}"
    ax.set_title(f'{model_name} Cumulative Returns ({date_range})', 
                fontsize=16, fontweight='bold')
    
    # Add grid with minor lines
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.1, which='minor', linestyle=':')
    ax.minorticks_on()
    
    # Format dates
    format_axis_dates(ax, 'quarter' if len(cum_returns) > 500 else 'month')
    
    # Improved legend positioning
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9)
    
    # Add key metrics annotation
    final_return = cum_returns.iloc[-1]
    max_return = cum_returns.max()
    min_return = cum_returns.min()
    
    metrics_text = f'Final: {final_return:.1f}%\nMax: {max_return:.1f}%\nMin: {min_return:.1f}%'
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=color, alpha=0.9),
            fontsize=10, ha='right', va='bottom')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_drawdown(nav: pd.Series,
                 model_name: str,
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced drawdown chart with duration information
    
    Args:
        nav: NAV series
        model_name: Name of the model
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Calculate drawdown
    running_max = nav.expanding().max()
    drawdown = ((nav - running_max) / running_max) * 100
    
    # Color coding by severity
    color = MODEL_COLORS.get(model_name, '#333333')
    
    # Main drawdown plot
    ax1.fill_between(drawdown.index, 0, drawdown.values, 
                     where=(drawdown.values > -10), color='yellow', 
                     alpha=0.3, label='Mild (<10%)', interpolate=True)
    ax1.fill_between(drawdown.index, 0, drawdown.values, 
                     where=(drawdown.values <= -10) & (drawdown.values > -20), 
                     color='orange', alpha=0.4, label='Moderate (10-20%)', interpolate=True)
    ax1.fill_between(drawdown.index, 0, drawdown.values, 
                     where=(drawdown.values <= -20), color='red', 
                     alpha=0.5, label='Severe (>20%)', interpolate=True)
    
    ax1.plot(drawdown.index, drawdown.values, 
            color=color, linewidth=1.5, alpha=0.8)
    
    # Mark max drawdown with better positioning
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax1.plot(max_dd_date, max_dd, 'ro', markersize=8)
    
    # Smart positioning of max drawdown label
    y_pos = max_dd - 2 if max_dd > -25 else max_dd + 5
    ax1.annotate(f'Max: {max_dd:.1f}%',
                xy=(max_dd_date, max_dd),
                xytext=(max_dd_date, y_pos),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='red', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # Formatting
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Drawdown (%)', fontsize=14)
    ax1.set_title(f'{model_name} Underwater Plot', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Duration plot
    drawdown_periods = []
    in_dd = False
    start_dd = None
    
    for date, dd in drawdown.items():
        if dd < 0 and not in_dd:
            start_dd = date
            in_dd = True
        elif dd >= 0 and in_dd:
            duration = (date - start_dd).days
            drawdown_periods.append({
                'start': start_dd,
                'end': date,
                'duration': duration,
                'max_dd': drawdown[start_dd:date].min()
            })
            in_dd = False
    
    if drawdown_periods:
        # Plot duration bars
        durations = pd.DataFrame(drawdown_periods)
        colors = ['yellow' if dd > -10 else 'orange' if dd > -20 else 'red' 
                 for dd in durations['max_dd']]
        
        ax2.bar(range(len(durations)), durations['duration'], color=colors, alpha=0.6)
        ax2.set_ylabel('Duration (days)', fontsize=12)
        ax2.set_xlabel('Drawdown Episode', fontsize=12)
        ax2.set_title('Drawdown Durations', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add average duration line
        avg_duration = durations['duration'].mean()
        ax2.axhline(y=avg_duration, color='red', linestyle='--', 
                   alpha=0.7, label=f'Avg: {avg_duration:.0f} days')
        ax2.legend(loc='upper right')
    
    # Format dates on main plot
    format_axis_dates(ax1, 'quarter' if len(drawdown) > 500 else 'month')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_monthly_heatmap(nav: pd.Series,
                         model_name: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced monthly returns heatmap with yearly totals and centered colormap
    
    Args:
        nav: NAV series
        model_name: Name of the model
        save_path: Path to save figure
    """
    set_publication_style('clean')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate monthly returns
    monthly_nav = nav.resample('M').last()
    monthly_returns = monthly_nav.pct_change().dropna() * 100
    
    # Create matrix (years as rows, months as columns)
    years = sorted(monthly_returns.index.year.unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Add yearly total column
    months_with_total = months + ['Year']
    returns_matrix = pd.DataFrame(index=years, columns=months_with_total)
    
    # Fill monthly returns
    for date, ret in monthly_returns.items():
        returns_matrix.loc[date.year, months[date.month - 1]] = ret
    
    # Calculate yearly returns
    for year in years:
        year_returns = returns_matrix.loc[year, months].dropna()
        if len(year_returns) > 0:
            # Compound monthly returns for yearly total
            yearly_return = ((1 + year_returns / 100).prod() - 1) * 100
            returns_matrix.loc[year, 'Year'] = yearly_return
    
    # Add average row
    avg_row = pd.DataFrame(index=['Avg'], columns=months_with_total)
    for month in months:
        month_values = returns_matrix[month].dropna()
        if len(month_values) > 0:
            avg_row.loc['Avg', month] = month_values.mean()
    
    # Average of yearly returns
    year_values = returns_matrix['Year'].dropna()
    if len(year_values) > 0:
        avg_row.loc['Avg', 'Year'] = year_values.mean()
    
    # Combine with averages
    plot_matrix = pd.concat([returns_matrix, avg_row])
    
    # Separate data for heatmap (exclude Year column for color scaling)
    heatmap_data = plot_matrix[months].astype(float)
    
    # Determine color scale bounds (centered at zero)
    vmax = max(abs(heatmap_data.min().min()), heatmap_data.max().max())
    vmin = -vmax
    
    # Create mask for missing data
    mask = plot_matrix.isna()
    
    # Plot heatmap
    sns.heatmap(plot_matrix.astype(float), 
               annot=True, fmt='.1f', 
               cmap='RdYlGn', center=0,
               cbar_kws={'label': 'Return (%)'},
               vmin=vmin, vmax=vmax,
               linewidths=1, linecolor='gray',
               mask=mask,
               ax=ax)
    
    # Highlight best and worst months
    best_month = heatmap_data.max().max()
    worst_month = heatmap_data.min().min()
    
    # Add markers for best/worst (this is complex with seaborn, so we'll add text)
    for i, year in enumerate(plot_matrix.index):
        for j, month in enumerate(months):
            value = plot_matrix.loc[year, month]
            if pd.notna(value):
                if value == best_month:
                    ax.text(j + 0.5, i + 0.2, '★', ha='center', va='center', 
                           color='gold', fontsize=16, fontweight='bold')
                elif value == worst_month:
                    ax.text(j + 0.5, i + 0.8, '▼', ha='center', va='center', 
                           color='darkred', fontsize=12, fontweight='bold')
    
    # Draw vertical line before Year column
    ax.axvline(x=12, color='black', linewidth=2)
    
    # Draw horizontal line before Avg row
    if 'Avg' in plot_matrix.index:
        avg_position = len(plot_matrix.index) - 1
        ax.axhline(y=avg_position, color='black', linewidth=2)
    
    # Formatting
    ax.set_title(f'{model_name} Monthly Returns Heatmap', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('Year', fontsize=14)
    
    # Rotate labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    
    # Add legend for markers
    legend_text = '★ Best Month  ▼ Worst Month'
    ax.text(0.99, 0.01, legend_text, transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_return_distribution(nav: pd.Series,
                            model_name: str,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced return distribution with KDE overlay and statistics
    
    Args:
        nav: NAV series
        model_name: Name of the model
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate returns
    returns = nav.pct_change().dropna() * 100
    
    color = MODEL_COLORS.get(model_name, '#333333')
    
    # Left panel: Enhanced histogram with KDE
    n, bins, patches = ax1.hist(returns, bins=50, density=True, 
                                alpha=0.6, color=color, edgecolor='black', linewidth=0.5)
    
    # Add KDE overlay
    kde = stats.gaussian_kde(returns)
    x_range = np.linspace(returns.min(), returns.max(), 200)
    kde_values = kde(x_range)
    ax1.plot(x_range, kde_values, color=color, linewidth=2.5, label='KDE')
    
    # Fit normal distribution
    mu, std = returns.mean(), returns.std()
    normal_dist = stats.norm(mu, std)
    normal_values = normal_dist.pdf(x_range)
    ax1.plot(x_range, normal_values, 'k--', linewidth=2, alpha=0.7, label='Normal fit')
    
    # Add vertical lines for key statistics
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean: {mu:.2f}%')
    ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Median: {returns.median():.2f}%')
    
    # Add percentile lines
    p5, p95 = np.percentile(returns, [5, 95])
    ax1.axvline(p5, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'5th %ile: {p5:.2f}%')
    ax1.axvline(p95, color='orange', linestyle=':', linewidth=1.5, 
               alpha=0.7, label=f'95th %ile: {p95:.2f}%')
    
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Return Distribution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Enhanced Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(color)
    ax2.get_lines()[0].set_markeredgecolor('black')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2)
    
    ax2.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax2.set_ylabel('Sample Quantiles', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    skew = returns.skew()
    kurt = returns.kurtosis()
    sharpe = mu / std * np.sqrt(252)  # Annualized
    
    stats_text = (f'Statistics:\n'
                 f'Mean: {mu:.3f}%\n'
                 f'Std: {std:.3f}%\n'
                 f'Skew: {skew:.3f}\n'
                 f'Kurtosis: {kurt:.3f}\n'
                 f'Sharpe: {sharpe:.3f}')
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='gray', alpha=0.9),
            fontsize=10, ha='left', va='top')
    
    plt.suptitle(f'{model_name} Return Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_sharpe(nav: pd.Series,
                       model_name: str,
                       window: int = 252,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced rolling Sharpe ratio with smoothed trend
    
    Args:
        nav: NAV series
        model_name: Name of the model
        window: Rolling window size
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Calculate returns
    returns = nav.pct_change().dropna()
    
    # Calculate rolling Sharpe
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    color = MODEL_COLORS.get(model_name, '#333333')
    
    # Plot rolling Sharpe
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
            color=color, linewidth=1.5, alpha=0.7, label=f'{window}-day Rolling')
    
    # Add smoothed trend
    if len(rolling_sharpe.dropna()) > 30:
        smoothed = add_smooth_trend(ax, rolling_sharpe.index, rolling_sharpe.values,
                                   window=30, label='30-day MA', 
                                   color='red', alpha=0.8)
    
    # Add horizontal lines for reference
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
    ax.axhline(y=-1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = -1')
    
    # Add average line
    avg_sharpe = rolling_sharpe.mean()
    ax.axhline(y=avg_sharpe, color='blue', linestyle=':', linewidth=2, 
              alpha=0.7, label=f'Average: {avg_sharpe:.2f}')
    
    # Highlight periods of strong performance
    strong_perf = rolling_sharpe > 1.5
    if strong_perf.any():
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                       where=strong_perf, color='green', alpha=0.1, 
                       label='Strong Performance (>1.5)')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sharpe Ratio', fontsize=14)
    ax.set_title(f'{model_name} Rolling Sharpe Ratio ({window}-day)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter' if len(rolling_sharpe) > 500 else 'month')
    
    # Add current Sharpe annotation
    current_sharpe = rolling_sharpe.iloc[-1]
    if not np.isnan(current_sharpe):
        ax.annotate(f'Current: {current_sharpe:.2f}',
                   xy=(rolling_sharpe.index[-1], current_sharpe),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8),
                   fontsize=11, ha='left')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_rolling_volatility(nav: pd.Series,
                           model_name: str,
                           window: int = 60,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced rolling volatility with regime bands
    
    Args:
        nav: NAV series
        model_name: Name of the model
        window: Rolling window size
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
    
    # Calculate returns and volatility
    returns = nav.pct_change().dropna()
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    
    color = MODEL_COLORS.get(model_name, '#333333')
    
    # Plot rolling volatility
    ax.plot(rolling_vol.index, rolling_vol.values, 
            color=color, linewidth=2, label=f'{window}-day Rolling Vol')
    
    # Add volatility regime bands
    vol_percentiles = rolling_vol.quantile([0.25, 0.5, 0.75])
    
    ax.axhspan(0, vol_percentiles[0.25], alpha=0.1, color='green', label='Low Vol')
    ax.axhspan(vol_percentiles[0.25], vol_percentiles[0.75], alpha=0.1, color='yellow', label='Normal Vol')
    ax.axhspan(vol_percentiles[0.75], rolling_vol.max() * 1.1, alpha=0.1, color='red', label='High Vol')
    
    # Add median line
    ax.axhline(y=vol_percentiles[0.5], color='blue', linestyle='--', 
              linewidth=1.5, alpha=0.7, label=f'Median: {vol_percentiles[0.5]:.1f}%')
    
    # Mark volatility spikes
    vol_spikes = rolling_vol > rolling_vol.quantile(0.95)
    spike_dates = rolling_vol[vol_spikes].index
    
    for date in spike_dates[:10]:  # Mark top 10 spikes
        ax.axvline(x=date, color='red', alpha=0.3, linestyle=':', linewidth=1)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=14)
    ax.set_title(f'{model_name} Rolling Volatility ({window}-day)', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Format dates
    format_axis_dates(ax, 'quarter' if len(rolling_vol) > 500 else 'month')
    
    # Add current volatility annotation
    current_vol = rolling_vol.iloc[-1]
    if not np.isnan(current_vol):
        ax.annotate(f'Current: {current_vol:.1f}%',
                   xy=(rolling_vol.index[-1], current_vol),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.8),
                   fontsize=11, ha='left')
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def create_all_individual_plots(nav: pd.Series,
                               model_name: str,
                               benchmark_nav: Optional[pd.Series] = None,
                               base_path: str = './plots/individual/') -> Dict[str, plt.Figure]:
    """
    Create all individual plots for a model
    
    Args:
        nav: NAV series
        model_name: Name of the model
        benchmark_nav: Optional benchmark NAV
        base_path: Base path for saving plots
    
    Returns:
        Dictionary of figure objects
    """
    import os
    
    # Create directory
    save_dir = os.path.join(base_path, model_name.lower().replace(' ', '_'))
    os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # Create all plots
    figures['cumulative_returns'] = plot_cumulative_returns(
        nav, model_name, benchmark_nav,
        save_path=os.path.join(save_dir, 'cumulative_returns.png')
    )
    
    figures['drawdown'] = plot_drawdown(
        nav, model_name,
        save_path=os.path.join(save_dir, 'drawdown.png')
    )
    
    figures['monthly_heatmap'] = plot_monthly_heatmap(
        nav, model_name,
        save_path=os.path.join(save_dir, 'monthly_heatmap.png')
    )
    
    figures['return_distribution'] = plot_return_distribution(
        nav, model_name,
        save_path=os.path.join(save_dir, 'return_distribution.png')
    )
    
    figures['rolling_sharpe'] = plot_rolling_sharpe(
        nav, model_name,
        save_path=os.path.join(save_dir, 'rolling_sharpe.png')
    )
    
    figures['rolling_volatility'] = plot_rolling_volatility(
        nav, model_name,
        save_path=os.path.join(save_dir, 'rolling_volatility.png')
    )
    
    print(f"✓ Created all individual plots for {model_name} in {save_dir}")
    
    return figures