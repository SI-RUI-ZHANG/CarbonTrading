"""
Simplified visualization module for backtesting results
Each plot has a single, clear purpose with minimal overlapping information
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Union
from scipy import stats
from plot_config import *

# Base directory for plots
PLOTS_BASE_DIR = '../../Plots/backtesting/'

def ensure_dir_exists(filepath: str):
    """Ensure directory exists for the given filepath"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

# ============================================================================
# INDIVIDUAL MODEL ANALYSIS - Detailed plots for single models
# ============================================================================

def plot_individual_analysis(nav: pd.Series,
                            model_name: str,
                            market: str,
                            frequency: str,
                            benchmark_nav: Optional[pd.Series] = None,
                            save_dir: Optional[str] = None):
    """
    Generate all individual analysis plots for a single model
    
    Args:
        nav: NAV series for the model
        model_name: Name of the model
        market: Market name (GDEA/HBEA)
        frequency: Frequency (daily/weekly)
        benchmark_nav: Optional Buy&Hold benchmark
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = f"{PLOTS_BASE_DIR}individual/{frequency}_{market}_{model_name.lower()}/"
    
    # 1. Cumulative returns
    plot_cumulative_returns_single(
        nav, model_name, benchmark_nav,
        save_path=f"{save_dir}cumulative_returns.png"
    )
    
    # 2. Drawdown analysis
    plot_drawdown_single(
        nav, model_name,
        save_path=f"{save_dir}drawdown.png"
    )
    
    # 3. Monthly heatmap
    plot_monthly_heatmap_single(
        nav, model_name,
        save_path=f"{save_dir}monthly_heatmap.png"
    )
    
    # 4. Return distribution
    plot_return_distribution_single(
        nav, model_name,
        save_path=f"{save_dir}return_distribution.png"
    )

def plot_cumulative_returns_single(nav: pd.Series,
                                  model_name: str,
                                  benchmark_nav: Optional[pd.Series] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot cumulative returns for a single model with optional benchmark
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative returns
    cum_returns = (nav / nav.iloc[0] - 1) * 100
    
    # Plot main model
    color = MODEL_COLORS.get(model_name, '#2E7D32')
    ax.plot(cum_returns.index, cum_returns.values,
           color=color, linewidth=2.5, label=model_name)
    ax.fill_between(cum_returns.index, 0, cum_returns.values,
                   alpha=0.15, color=color)
    
    # Add benchmark if provided
    if benchmark_nav is not None:
        bench_returns = (benchmark_nav / benchmark_nav.iloc[0] - 1) * 100
        ax.plot(bench_returns.index, bench_returns.values,
               color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Buy&Hold')
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add final return annotation
    final_return = cum_returns.iloc[-1]
    ax.text(0.02, 0.98, f'Final Return: {final_return:.1f}%',
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=color, alpha=0.9))
    
    # Format x-axis dates
    format_axis_dates(ax, 'quarter')
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_drawdown_single(nav: pd.Series,
                        model_name: str,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot drawdown analysis for a single model
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate drawdown
    running_max = nav.expanding().max()
    drawdown = ((nav - running_max) / running_max) * 100
    
    # Plot drawdown
    ax.fill_between(drawdown.index, 0, drawdown.values,
                   color='red', alpha=0.4, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values,
           color='darkred', linewidth=1.5)
    
    # Mark maximum drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.scatter([max_dd_date], [max_dd], color='red', s=100, zorder=5)
    ax.annotate(f'Max: {max_dd:.1f}%\n{max_dd_date.strftime("%Y-%m-%d")}',
               xy=(max_dd_date, max_dd),
               xytext=(10, -10), textcoords='offset points',
               fontsize=10, color='darkred',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='red', alpha=0.9))
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-10, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=-20, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} Drawdown Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Format x-axis dates
    format_axis_dates(ax, 'quarter')
    
    # Set y-axis limits
    ax.set_ylim([max_dd * 1.2, 5])
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_monthly_heatmap_single(nav: pd.Series,
                               model_name: str,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot monthly returns heatmap for a single model
    """
    set_publication_style('clean')
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate monthly returns
    monthly_nav = nav.resample('ME').last()
    monthly_returns = monthly_nav.pct_change().dropna() * 100
    
    # Create matrix
    years = sorted(monthly_returns.index.year.unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    returns_matrix = pd.DataFrame(index=years, columns=months)
    
    for date, ret in monthly_returns.items():
        returns_matrix.loc[date.year, months[date.month - 1]] = ret
    
    # Calculate yearly returns
    yearly_returns = []
    for year in years:
        year_data = returns_matrix.loc[year].dropna()
        if len(year_data) > 0:
            # Compound monthly returns
            year_return = ((1 + year_data / 100).prod() - 1) * 100
            yearly_returns.append(year_return)
        else:
            yearly_returns.append(np.nan)
    
    # Plot heatmap
    vmax = max(abs(returns_matrix.min().min()), abs(returns_matrix.max().max()))
    vmin = -vmax
    
    sns.heatmap(returns_matrix.astype(float),
               annot=True, fmt='.1f',
               cmap='RdYlGn', center=0,
               vmin=vmin, vmax=vmax,
               cbar_kws={'label': 'Monthly Return (%)'},
               linewidths=0.5, linecolor='gray',
               ax=ax)
    
    # Add yearly returns as text
    for i, (year, ret) in enumerate(zip(years, yearly_returns)):
        if not np.isnan(ret):
            ax.text(12.5, i + 0.5, f'{ret:.1f}%',
                   fontsize=10, fontweight='bold',
                   ha='left', va='center')
    
    ax.text(12.5, -0.5, 'Year', fontsize=10, fontweight='bold', ha='left')
    
    # Formatting
    ax.set_title(f'{model_name} Monthly Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Year', fontsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_return_distribution_single(nav: pd.Series,
                                   model_name: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot return distribution analysis for a single model
    """
    set_publication_style('grid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate returns
    returns = nav.pct_change().dropna() * 100
    
    # Left panel: Histogram with KDE
    color = MODEL_COLORS.get(model_name, '#2E7D32')
    n, bins, patches = ax1.hist(returns, bins=50, density=True,
                               alpha=0.6, color=color,
                               edgecolor='black', linewidth=0.5)
    
    # Add KDE
    kde = stats.gaussian_kde(returns)
    x_range = np.linspace(returns.min(), returns.max(), 200)
    ax1.plot(x_range, kde(x_range), color=color, linewidth=2, label='KDE')
    
    # Add normal fit
    mu, std = returns.mean(), returns.std()
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    p_norm = stats.norm.pdf(x_norm, mu, std)
    ax1.plot(x_norm, p_norm, 'r--', linewidth=1.5, alpha=0.7, label='Normal')
    
    # Add vertical lines
    ax1.axvline(mu, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=1, alpha=0.7)
    
    ax1.set_xlabel('Daily Return (%)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(color)
    ax2.get_lines()[0].set_markeredgecolor('black')
    ax2.get_lines()[0].set_markersize(4)
    ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics box
    textstr = f'Mean: {mu:.2f}%\n'
    textstr += f'Std: {std:.2f}%\n'
    textstr += f'Skew: {returns.skew():.2f}\n'
    textstr += f'Kurt: {returns.kurtosis():.2f}\n'
    textstr += f'VaR(95%): {np.percentile(returns, 5):.2f}%'
    
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    fig.suptitle(f'{model_name} Return Distribution Analysis',
                fontsize=14, fontweight='bold')
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

# ============================================================================
# SIMPLE COMPARISONS - Focused cross-model comparisons that add value
# ============================================================================

def plot_cumulative_returns(nav_dict: Dict[str, pd.Series],
                           title: str = "Cumulative Returns Comparison",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Simple cumulative returns comparison (max 3 models for clarity)
    
    Args:
        nav_dict: Dictionary of NAV series (max 3 recommended)
        title: Plot title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Limit to 3 models for clarity
    if len(nav_dict) > 3:
        # Keep Buy&Hold and top 2 performers
        from c_evaluation_metrics import calculate_comprehensive_metrics
        metrics = {name: calculate_comprehensive_metrics(nav) for name, nav in nav_dict.items()}
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        selected = {}
        if 'Buy&Hold' in nav_dict:
            selected['Buy&Hold'] = nav_dict['Buy&Hold']
        for name, _ in sorted_models[:2]:
            if name != 'Buy&Hold':
                selected[name] = nav_dict[name]
        nav_dict = selected
    
    # Plot each model
    for name, nav in nav_dict.items():
        cum_returns = (nav / nav.iloc[0] - 1) * 100
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = '-' if name != 'Buy&Hold' else '--'
        linewidth = 2.5 if name != 'Buy&Hold' else 2
        alpha = 1.0 if name != 'Buy&Hold' else 0.7
        
        ax.plot(cum_returns.index, cum_returns.values,
               color=color, linestyle=linestyle, linewidth=linewidth,
               alpha=alpha, label=name)
    
    # Formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Format x-axis dates
    format_axis_dates(ax, 'quarter')
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_sharpe_comparison(metrics_dict: Dict[str, Dict],
                         market: str,
                         frequency: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Simple bar chart comparing Sharpe ratios across models
    """
    set_publication_style('clean')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    models = list(metrics_dict.keys())
    sharpes = [metrics_dict[m].get('sharpe_ratio', 0) for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    # Create bars
    bars = ax.bar(models, sharpes, color=colors, alpha=0.8,
                 edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, sharpes):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{value:.2f}', ha='center', va=va,
               fontsize=11, fontweight='bold')
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1,
              alpha=0.5, label='Good (SR=1)')
    
    # Formatting
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title(f'Sharpe Ratio Comparison - {market} {frequency.capitalize()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=9)
    
    # Set y-limits with padding
    if sharpes:
        y_min = min(0, min(sharpes) * 1.2)
        y_max = max(sharpes) * 1.2
        ax.set_ylim([y_min, y_max])
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_risk_return_scatter(metrics_dict: Dict[str, Dict],
                           market: str,
                           frequency: str,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Risk-return scatter plot showing efficiency
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    for name, metrics in metrics_dict.items():
        risk = metrics.get('annual_volatility', 0) * 100
        ret = metrics.get('cagr', 0) * 100
        color = MODEL_COLORS.get(name, '#333333')
        marker = 'o' if name == 'Buy&Hold' else 's'
        
        ax.scatter(risk, ret, color=color, s=200,
                  marker=marker, edgecolor='white',
                  linewidth=2, alpha=0.9, label=name)
        
        # Add label next to point
        ax.annotate(name, xy=(risk, ret),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Add Sharpe ratio reference lines
    if metrics_dict:
        max_risk = max([m.get('annual_volatility', 0) * 100 for m in metrics_dict.values()]) * 1.2
        for sharpe in [0.5, 1.0, 1.5]:
            x = np.linspace(0, max_risk, 100)
            y = sharpe * x
            ax.plot(x, y, '--', alpha=0.2, color='gray', linewidth=1)
            # Add label
            ax.text(max_risk * 0.9, sharpe * max_risk * 0.9,
                   f'SR={sharpe}', fontsize=8, alpha=0.5, rotation=45)
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Annual Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('CAGR (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Risk-Return Profile - {market} {frequency.capitalize()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', framealpha=0.9)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_model_comparison(nav_dict: Dict[str, pd.Series],
                         metrics_dict: Dict[str, Dict],
                         market: str,
                         frequency: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Simplified comparison focusing on key metrics only
    Creates a 2x2 grid with essential comparisons
    """
    set_publication_style('grid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cumulative Returns (top left)
    ax1 = axes[0, 0]
    for name, nav in nav_dict.items():
        cum_returns = (nav / nav.iloc[0] - 1) * 100
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = '-' if name != 'Buy&Hold' else '--'
        ax1.plot(cum_returns.index, cum_returns.values,
                label=name, color=color, linestyle=linestyle, linewidth=2)
    
    ax1.set_title('Cumulative Returns')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Return (%)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 2. Risk-Return Scatter (top right)
    ax2 = axes[0, 1]
    for name, metrics in metrics_dict.items():
        risk = metrics.get('annual_volatility', 0) * 100
        ret = metrics.get('cagr', 0) * 100
        color = MODEL_COLORS.get(name, '#333333')
        ax2.scatter(risk, ret, s=150, label=name, color=color, alpha=0.8)
        ax2.annotate(name, (risk, ret), xytext=(3, 3),
                    textcoords='offset points', fontsize=8)
    
    ax2.set_title('Risk-Return Profile')
    ax2.set_xlabel('Volatility (%)')
    ax2.set_ylabel('CAGR (%)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 3. Sharpe Ratio Bars (bottom left)
    ax3 = axes[1, 0]
    models = list(metrics_dict.keys())
    sharpes = [metrics_dict[m].get('sharpe_ratio', 0) for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    bars = ax3.bar(models, sharpes, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, sharpes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax3.set_title('Sharpe Ratio')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Maximum Drawdown Bars (bottom right)
    ax4 = axes[1, 1]
    drawdowns = [metrics_dict[m].get('max_drawdown', 0) * 100 for m in models]
    bars = ax4.bar(models, drawdowns, color=colors, alpha=0.8)
    
    for bar, value in zip(bars, drawdowns):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height - 1,
                f'{value:.1f}%', ha='center', va='top', fontsize=9)
    
    ax4.set_title('Maximum Drawdown')
    ax4.set_ylabel('Drawdown (%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'{market} {frequency.capitalize()} Model Comparison',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_performance_panels(nav: pd.Series,
                           metrics: Dict,
                           model_name: str,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Simplified performance panel - just key metrics for individual model
    """
    # Redirect to individual analysis
    market = "Unknown"
    frequency = "daily"
    
    # Extract from model name if possible
    if "_" in model_name:
        parts = model_name.split("_")
        if len(parts) >= 2:
            market = parts[-1]
            frequency = parts[0] if parts[0] in ['daily', 'weekly'] else 'daily'
    
    plot_individual_analysis(nav, model_name, market, frequency, save_dir=save_path)
    
    # Return a dummy figure for compatibility
    fig = plt.figure()
    return fig

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_summary_latex_table(metrics_dict: Dict[str, Dict],
                              market: str,
                              frequency: str,
                              save_path: Optional[str] = None) -> str:
    """
    Create LaTeX table for paper
    """
    latex = []
    latex.append(r'\begin{table}[h]')
    latex.append(r'\centering')
    latex.append(r'\caption{' + f'{market} {frequency.capitalize()} Performance Metrics' + r'}')
    latex.append(r'\begin{tabular}{lrrrrr}')
    latex.append(r'\hline')
    latex.append(r'Model & Return (\%) & CAGR (\%) & Sharpe & Max DD (\%) & Win Rate (\%) \\')
    latex.append(r'\hline')
    
    for name, metrics in metrics_dict.items():
        total_ret = metrics.get('total_return', 0) * 100
        cagr = metrics.get('cagr', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0) * 100
        win_rate = metrics.get('positive_days', 0) * 100
        
        latex.append(f'{name} & {total_ret:.1f} & {cagr:.1f} & {sharpe:.2f} & {max_dd:.1f} & {win_rate:.1f} \\\\')
    
    latex.append(r'\hline')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')
    
    latex_str = '\n'.join(latex)
    
    if save_path:
        ensure_dir_exists(save_path)
        with open(save_path, 'w') as f:
            f.write(latex_str)
    
    return latex_str