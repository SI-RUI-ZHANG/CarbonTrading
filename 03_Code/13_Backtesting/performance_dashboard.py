"""
Performance Summary Dashboard
Creates a comprehensive single-page view of all key metrics and visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional
import seaborn as sns
from plot_config import *

def ensure_dir_exists(filepath: str):
    """Ensure directory exists for the given filepath"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

def create_performance_dashboard(nav_dict: Dict[str, pd.Series],
                                metrics_dict: Dict[str, Dict],
                                market: str,
                                frequency: str,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive performance dashboard
    
    Args:
        nav_dict: Dictionary of NAV series for each model
        metrics_dict: Dictionary of metrics for each model
        market: Market name (GDEA/HBEA)
        frequency: Trading frequency (daily/weekly)
        save_path: Path to save figure
    """
    set_publication_style('clean')
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'{market} {frequency.capitalize()} Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Cumulative Returns (top left, 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_cumulative_returns_mini(ax1, nav_dict, f'{market} {frequency.capitalize()}')
    
    # 2. Risk-Return Scatter (top right, 2x2)
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    plot_risk_return_mini(ax2, metrics_dict)
    
    # 3. Metrics Table (middle left, 1x2)
    ax3 = fig.add_subplot(gs[2, 0:2])
    plot_metrics_table_mini(ax3, metrics_dict)
    
    # 4. Sharpe Comparison (middle right, 1x1)
    ax4 = fig.add_subplot(gs[2, 2])
    plot_sharpe_bars_mini(ax4, metrics_dict)
    
    # 5. Drawdown Comparison (middle right, 1x1)
    ax5 = fig.add_subplot(gs[2, 3])
    plot_drawdown_bars_mini(ax5, metrics_dict)
    
    # 6. Monthly Heatmap (bottom, 1x3)
    ax6 = fig.add_subplot(gs[3, 0:3])
    best_model = get_best_model(metrics_dict)
    if best_model in nav_dict:
        plot_monthly_heatmap_mini(ax6, nav_dict[best_model], best_model)
    
    # 7. Win Rate Stats (bottom right, 1x1)
    ax7 = fig.add_subplot(gs[3, 3])
    plot_win_rate_mini(ax7, nav_dict, metrics_dict)
    
    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', 
            fontsize=8, ha='right', va='bottom', alpha=0.5)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path, dpi=300)
    
    return fig

def plot_cumulative_returns_mini(ax, nav_dict, title):
    """Mini version of cumulative returns for dashboard"""
    for name, nav in nav_dict.items():
        cum_returns = (nav / nav.iloc[0] - 1) * 100
        color = MODEL_COLORS.get(name, '#333333')
        linestyle = MODEL_LINESTYLES.get(name, '-')
        alpha = MODEL_ALPHAS.get(name, 1.0)
        
        ax.plot(cum_returns.index, cum_returns.values,
               color=color, linestyle=linestyle, linewidth=2,
               alpha=alpha, label=name)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Return (%)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    format_axis_dates(ax, 'quarter')

def plot_risk_return_mini(ax, metrics_dict):
    """Mini version of risk-return scatter for dashboard"""
    for name, metrics in metrics_dict.items():
        risk = metrics.get('annual_volatility', 0) * 100
        ret = metrics.get('cagr', 0) * 100
        color = MODEL_COLORS.get(name, '#333333')
        
        ax.scatter(risk, ret, color=color, s=150,
                  edgecolor='white', linewidth=2,
                  alpha=0.9, label=name)
        
        # Add label next to point
        ax.annotate(name, (risk, ret), xytext=(3, 3),
                   textcoords='offset points', fontsize=8)
    
    # Add Sharpe ratio lines
    max_risk = max([m.get('annual_volatility', 0) * 100 for m in metrics_dict.values()]) * 1.2
    for sharpe in [0.5, 1.0, 1.5]:
        x = np.linspace(0, max_risk, 100)
        y = sharpe * x
        ax.plot(x, y, '--', alpha=0.2, color='gray', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    ax.set_xlabel('Volatility (%)', fontsize=10)
    ax.set_ylabel('CAGR (%)', fontsize=10)
    ax.grid(True, alpha=0.3)

def plot_metrics_table_mini(ax, metrics_dict):
    """Mini metrics table for dashboard"""
    ax.axis('off')
    
    # Select key metrics
    metrics_to_show = [
        ('total_return', 'Total Ret'),
        ('cagr', 'CAGR'),
        ('sharpe_ratio', 'Sharpe'),
        ('max_drawdown', 'Max DD'),
        ('positive_days', 'Win Rate')
    ]
    
    # Create table data
    columns = ['Model'] + [m[1] for m in metrics_to_show]
    rows = []
    
    for name, metrics in metrics_dict.items():
        row = [name]
        for key, _ in metrics_to_show:
            if key in ['total_return', 'cagr', 'max_drawdown', 'positive_days']:
                value = metrics.get(key, 0) * 100
                row.append(f'{value:.1f}%')
            else:
                value = metrics.get(key, 0)
                row.append(f'{value:.2f}')
        rows.append(row)
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color model names
    for i, row in enumerate(rows, 1):
        model_name = row[0]
        color = MODEL_COLORS.get(model_name, '#FFFFFF')
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_alpha(0.3)
    
    ax.set_title('Key Metrics', fontsize=12, fontweight='bold', pad=20)

def plot_sharpe_bars_mini(ax, metrics_dict):
    """Mini Sharpe ratio bars for dashboard"""
    models = list(metrics_dict.keys())
    sharpes = [metrics_dict[m].get('sharpe_ratio', 0) for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    bars = ax.bar(range(len(models)), sharpes, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, sharpes):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{value:.2f}', ha='center', va=va, fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

def plot_drawdown_bars_mini(ax, metrics_dict):
    """Mini drawdown bars for dashboard"""
    models = list(metrics_dict.keys())
    dds = [metrics_dict[m].get('max_drawdown', 0) * 100 for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    bars = ax.bar(range(len(models)), dds, color=colors, alpha=0.8)
    
    # Color by severity
    for bar, value in zip(bars, dds):
        if value < -30:
            bar.set_facecolor('red')
        elif value < -20:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('yellow')
        bar.set_alpha(0.7)
        
        # Add value label
        ax.text(bar.get_x() + bar.get_width()/2., value - 1,
               f'{value:.1f}%', ha='center', va='top', fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Max Drawdown', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

def plot_monthly_heatmap_mini(ax, nav, model_name):
    """Mini monthly heatmap for dashboard"""
    # Calculate monthly returns
    monthly_nav = nav.resample('M').last()
    monthly_returns = monthly_nav.pct_change().dropna() * 100
    
    # Create matrix
    years = sorted(monthly_returns.index.year.unique())[-3:]  # Last 3 years
    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    returns_matrix = pd.DataFrame(index=years, columns=months)
    
    for date, ret in monthly_returns.items():
        if date.year in years:
            returns_matrix.loc[date.year, months[date.month - 1]] = ret
    
    # Plot heatmap
    sns.heatmap(returns_matrix.astype(float), 
               annot=True, fmt='.0f', 
               cmap='RdYlGn', center=0,
               vmin=-15, vmax=15,
               linewidths=0.5, linecolor='gray',
               cbar_kws={'label': 'Return (%)'},
               ax=ax)
    
    ax.set_title(f'{model_name} Monthly Returns (Last 3 Years)', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Year', fontsize=10)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

def plot_win_rate_mini(ax, nav_dict, metrics_dict):
    """Mini win rate plot for dashboard"""
    models = list(metrics_dict.keys())
    win_rates = []
    avg_wins = []
    avg_losses = []
    
    for model in models:
        win_rate = metrics_dict[model].get('positive_days', 0) * 100
        win_rates.append(win_rate)
        
        if model in nav_dict:
            returns = nav_dict[model].pct_change().dropna() * 100
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            avg_wins.append(wins.mean() if len(wins) > 0 else 0)
            avg_losses.append(abs(losses.mean()) if len(losses) > 0 else 0)
        else:
            avg_wins.append(0)
            avg_losses.append(0)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, win_rates, width, label='Win Rate', color='blue', alpha=0.7)
    ax.bar(x, avg_wins, width, label='Avg Win', color='green', alpha=0.7)
    ax.bar(x + width, avg_losses, width, label='Avg Loss', color='red', alpha=0.7)
    
    ax.set_title('Win/Loss Analysis', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percent (%)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

def get_best_model(metrics_dict):
    """Get the best model based on Sharpe ratio"""
    best_model = None
    best_sharpe = -float('inf')
    
    for name, metrics in metrics_dict.items():
        if name != 'Buy&Hold':  # Exclude benchmark
            sharpe = metrics.get('sharpe_ratio', -float('inf'))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model = name
    
    return best_model or list(metrics_dict.keys())[0]

def create_market_comparison_dashboard(daily_data: Dict, weekly_data: Dict,
                                      market: str,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a dashboard comparing daily and weekly performance for a market
    
    Args:
        daily_data: Dictionary with 'nav_dict' and 'metrics_dict' for daily
        weekly_data: Dictionary with 'nav_dict' and 'metrics_dict' for weekly
        market: Market name (GDEA/HBEA)
        save_path: Path to save figure
    """
    set_publication_style('clean')
    
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'{market} Daily vs Weekly Performance Comparison', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Daily cumulative returns (top left)
    ax1 = fig.add_subplot(gs[0, 0:2])
    plot_cumulative_returns_mini(ax1, daily_data['nav_dict'], 'Daily')
    ax1.set_title('Daily Cumulative Returns', fontsize=12, fontweight='bold')
    
    # Weekly cumulative returns (top right)
    ax2 = fig.add_subplot(gs[0, 2:4])
    plot_cumulative_returns_mini(ax2, weekly_data['nav_dict'], 'Weekly')
    ax2.set_title('Weekly Cumulative Returns', fontsize=12, fontweight='bold')
    
    # Daily risk-return (middle left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_risk_return_mini(ax3, daily_data['metrics_dict'])
    ax3.set_title('Daily Risk-Return', fontsize=12, fontweight='bold')
    
    # Weekly risk-return (middle right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    plot_risk_return_mini(ax4, weekly_data['metrics_dict'])
    ax4.set_title('Weekly Risk-Return', fontsize=12, fontweight='bold')
    
    # Combined metrics comparison (bottom)
    ax5 = fig.add_subplot(gs[2, :])
    plot_frequency_comparison_table(ax5, daily_data['metrics_dict'], 
                                   weekly_data['metrics_dict'])
    
    # Add timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', 
            fontsize=8, ha='right', va='bottom', alpha=0.5)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path, dpi=300)
    
    return fig

def plot_frequency_comparison_table(ax, daily_metrics, weekly_metrics):
    """Create a comparison table between daily and weekly metrics"""
    ax.axis('off')
    
    # Metrics to compare
    metrics_to_show = [
        ('total_return', 'Total Return', '{:.1f}%', 100),
        ('cagr', 'CAGR', '{:.1f}%', 100),
        ('sharpe_ratio', 'Sharpe', '{:.2f}', 1),
        ('max_drawdown', 'Max DD', '{:.1f}%', 100),
        ('positive_days', 'Win Rate', '{:.1f}%', 100)
    ]
    
    # Create comparison data
    rows = []
    models = set(list(daily_metrics.keys()) + list(weekly_metrics.keys()))
    
    for model in sorted(models):
        if model in daily_metrics and model in weekly_metrics:
            row = [model]
            
            # Add daily metrics
            for key, _, fmt, mult in metrics_to_show:
                value = daily_metrics[model].get(key, 0) * mult
                row.append(fmt.format(value))
            
            # Add weekly metrics
            for key, _, fmt, mult in metrics_to_show:
                value = weekly_metrics[model].get(key, 0) * mult
                row.append(fmt.format(value))
            
            rows.append(row)
    
    # Create column headers
    metric_names = [m[1] for m in metrics_to_show]
    daily_cols = [f'D: {m}' for m in metric_names]
    weekly_cols = [f'W: {m}' for m in metric_names]
    columns = ['Model'] + daily_cols + weekly_cols
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        if i == 0:
            table[(0, i)].set_facecolor('#2E7D32')
        elif i <= len(daily_cols):
            table[(0, i)].set_facecolor('#1976D2')  # Blue for daily
        else:
            table[(0, i)].set_facecolor('#7B1FA2')  # Purple for weekly
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color model names
    for i, row in enumerate(rows, 1):
        model_name = row[0]
        color = MODEL_COLORS.get(model_name, '#FFFFFF')
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_alpha(0.3)
        table[(i, 0)].set_text_props(weight='bold')
    
    ax.set_title('Daily vs Weekly Performance Comparison', 
                fontsize=12, fontweight='bold', pad=20)