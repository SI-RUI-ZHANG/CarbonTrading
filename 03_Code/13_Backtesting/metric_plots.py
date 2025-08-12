"""
Improved metric visualization functions
Creates bar charts, scatter plots, and comparison tables with enhanced features and auto-directory creation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from plot_config import *

def ensure_dir_exists(filepath: str):
    """Ensure directory exists for the given filepath"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

def plot_sharpe_comparison(metrics_dict: Dict[str, Dict],
                          title_suffix: str = "",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced Sharpe ratio comparison bar chart with annotations
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['bar'])
    
    # Extract data
    models = list(metrics_dict.keys())
    sharpe_values = [metrics_dict[m].get('sharpe_ratio', 0) for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    # Create bars
    bars = ax.bar(models, sharpe_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, sharpe_values):
        height = bar.get_height()
        y_pos = height + 0.01 if height >= 0 else height - 0.05
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{value:.2f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    # Mark best performer
    if sharpe_values:
        best_idx = np.argmax(sharpe_values)
        ax.text(bars[best_idx].get_x() + bars[best_idx].get_width()/2., 
               sharpe_values[best_idx] + 0.08,
               'Best', ha='center', va='bottom', fontsize=12, 
               fontweight='bold', color='green')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (SR=1)')
    
    # Formatting
    ax.set_ylabel('Sharpe Ratio', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title(f'Sharpe Ratio Comparison{title_suffix}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits with padding
    if sharpe_values:
        y_min = min(0, min(sharpe_values) * 1.2)
        y_max = max(sharpe_values) * 1.2
        ax.set_ylim([y_min, y_max])
    
    # Add legend if reference line exists
    if sharpe_values and max(sharpe_values) > 0.8:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Rotate x-labels if many models
    if len(models) > 4:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_returns_comparison(metrics_dict: Dict[str, Dict],
                          return_types: List[str] = ['total_return', 'cagr'],
                          title_suffix: str = "",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced returns comparison with grouped bars
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        return_types: Types of returns to show
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(metrics_dict.keys())
    n_models = len(models)
    n_types = len(return_types)
    
    # Bar positioning
    x = np.arange(n_models)
    width = 0.35
    
    # Create grouped bars
    for i, ret_type in enumerate(return_types):
        values = [metrics_dict[m].get(ret_type, 0) * 100 for m in models]
        offset = (i - n_types/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=ret_type.replace('_', ' ').title(),
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            y_pos = height + 1 if height >= 0 else height - 3
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{value:.1f}%', ha='center', va=va, fontsize=9)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Return (%)', fontsize=14)
    ax.set_title(f'Returns Comparison{title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_risk_return_scatter(metrics_dict: Dict[str, Dict],
                           title_suffix: str = "",
                           add_frontier: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced risk-return scatter plot with better visibility
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        title_suffix: Additional text for title
        add_frontier: Whether to add efficient frontier line
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['scatter'])
    
    # Extract data
    risks = []
    returns = []
    names = []
    colors = []
    markers = []
    
    marker_styles = {'Buy&Hold': 'o', 'Base': 's', 'Sentiment': 'D', 'Meta': '^'}
    
    for name, metrics in metrics_dict.items():
        risks.append(metrics.get('annual_volatility', 0) * 100)
        returns.append(metrics.get('cagr', 0) * 100)
        names.append(name)
        colors.append(MODEL_COLORS.get(name, '#333333'))
        markers.append(marker_styles.get(name, 'o'))
    
    # Create scatter plot with larger, more visible markers
    for i, name in enumerate(names):
        ax.scatter(risks[i], returns[i], 
                  color=colors[i], s=300,  # Increased size
                  marker=markers[i],
                  edgecolor='white', linewidth=2,  # White edge for contrast
                  alpha=0.9, label=name, zorder=5)
    
    # Smart label placement to avoid overlaps
    for i, name in enumerate(names):
        # Use smart offset based on position
        offset_x = 10 if i % 2 == 0 else -10
        offset_y = 10 if i < len(names)/2 else -10
        
        ax.annotate(name, 
                   xy=(risks[i], returns[i]),
                   xytext=(offset_x, offset_y), 
                   textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   color=colors[i],
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=colors[i],
                           alpha=0.8))
    
    # Add efficient frontier (enhanced visibility)
    if add_frontier and len(risks) > 2:
        # Calculate efficient frontier points
        risk_return_pairs = list(zip(risks, returns))
        risk_return_pairs.sort(key=lambda x: x[0])  # Sort by risk
        
        # Find efficient points (max return for each risk level)
        efficient_points = []
        max_return_so_far = -float('inf')
        
        for risk, ret in risk_return_pairs:
            if ret > max_return_so_far:
                efficient_points.append((risk, ret))
                max_return_so_far = ret
        
        if len(efficient_points) > 1:
            eff_risks, eff_returns = zip(*efficient_points)
            ax.plot(eff_risks, eff_returns, 
                   'k-', linewidth=2.5, alpha=0.4, label='Efficient Frontier',
                   zorder=1)
            
            # Add shading below efficient frontier
            ax.fill_between(eff_risks, min(returns) - 5 if returns else -5, eff_returns,
                          alpha=0.05, color='green', zorder=0)
    
    # Add Sharpe ratio reference lines (more visible)
    max_risk = max(risks) * 1.2 if risks else 50
    max_return = max(returns) * 1.2 if returns else 20
    
    for sharpe in [0.5, 1.0, 1.5, 2.0]:
        x_sharpe = np.linspace(0, max_risk, 100)
        y_sharpe = sharpe * x_sharpe
        line = ax.plot(x_sharpe, y_sharpe, '--', alpha=0.3, linewidth=1.5,
                      color='gray', zorder=0)[0]
        
        # Add Sharpe ratio labels along the lines
        label_x = max_risk * 0.8
        label_y = sharpe * label_x
        if label_y <= max_return:
            angle = np.degrees(np.arctan(sharpe))
            ax.text(label_x, label_y, f'SR={sharpe}', 
                   fontsize=9, alpha=0.6, rotation=angle,
                   ha='center', va='bottom')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Annual Volatility (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CAGR (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Risk-Return Profile{title_suffix}', fontsize=16, fontweight='bold')
    
    # Set axis limits with padding
    x_min = min(0, min(risks) - 2) if risks else 0
    x_max = max(risks) * 1.15 if risks else 50
    y_min = min(returns) - 2 if returns else -5
    y_max = max(returns) * 1.15 if returns else 20
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    # Enhanced legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                      shadow=True, framealpha=0.95, 
                      title='Models', title_fontsize=11)
    legend.get_title().set_fontweight('bold')
    
    # Add annotation for interpretation
    interpretation = "↗ Better\n(Higher return,\nlower risk)"
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           fontsize=10, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                   edgecolor='green', alpha=0.3))
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path, dpi=300)
    
    return fig

def plot_drawdown_comparison(metrics_dict: Dict[str, Dict],
                           title_suffix: str = "",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced drawdown comparison bar chart
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, ax = plt.subplots(figsize=FIGURE_SIZES['bar'])
    
    # Extract data
    models = list(metrics_dict.keys())
    dd_values = [metrics_dict[m].get('max_drawdown', 0) * 100 for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    # Create bars (negative values)
    bars = ax.bar(models, dd_values, color=colors, alpha=0.8, 
                 edgecolor='black', linewidth=1.5)
    
    # Color bars by severity
    for bar, value in zip(bars, dd_values):
        if value < -30:
            bar.set_facecolor('red')
            bar.set_alpha(0.7)
        elif value < -20:
            bar.set_facecolor('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_facecolor('yellow')
            bar.set_alpha(0.7)
    
    # Add value labels
    for bar, value in zip(bars, dd_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 1,
               f'{value:.1f}%', ha='center', va='top', 
               fontsize=11, fontweight='bold')
    
    # Mark best (least negative)
    if dd_values:
        best_idx = np.argmax(dd_values)  # Least negative
        ax.text(bars[best_idx].get_x() + bars[best_idx].get_width()/2., 
               dd_values[best_idx] + 2,
               'Best', ha='center', va='bottom', fontsize=12, 
               fontweight='bold', color='green')
    
    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=-10, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='Mild')
    ax.axhline(y=-20, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate')
    ax.axhline(y=-30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Severe')
    
    # Formatting
    ax.set_ylabel('Maximum Drawdown (%)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title(f'Maximum Drawdown Comparison{title_suffix}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', framealpha=0.9)
    
    # Set y-axis limits
    if dd_values:
        y_min = min(dd_values) * 1.2
        ax.set_ylim([y_min, 5])
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_win_rate_comparison(metrics_dict: Dict[str, Dict],
                           nav_dict: Dict[str, pd.Series] = None,
                           title_suffix: str = "",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced win rate comparison with magnitude information
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        nav_dict: Dictionary of NAV series for calculating win/loss magnitudes
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('grid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(metrics_dict.keys())
    win_rates = [metrics_dict[m].get('positive_days', 0) * 100 for m in models]
    colors = [MODEL_COLORS.get(m, '#333333') for m in models]
    
    # Left panel: Win rate bars
    bars = ax1.bar(models, win_rates, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{value:.1f}%', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    # Add 50% reference line
    ax1.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_ylabel('Win Rate (%)', fontsize=14)
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_title('Daily Win Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(win_rates) * 1.2 if win_rates else 100])
    
    # Right panel: Average win vs average loss
    if nav_dict:
        avg_wins = []
        avg_losses = []
        
        for model in models:
            if model in nav_dict:
                returns = nav_dict[model].pct_change().dropna() * 100
                wins = returns[returns > 0]
                losses = returns[returns < 0]
                
                avg_wins.append(wins.mean() if len(wins) > 0 else 0)
                avg_losses.append(abs(losses.mean()) if len(losses) > 0 else 0)
            else:
                avg_wins.append(0)
                avg_losses.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, avg_wins, width, label='Avg Win', 
                       color='green', alpha=0.7, edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x + width/2, avg_losses, width, label='Avg Loss', 
                       color='red', alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                          f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # Calculate and display profit factor
        for i, model in enumerate(models):
            if avg_losses[i] > 0 and i < len(win_rates):
                profit_factor = (win_rates[i]/100 * avg_wins[i]) / ((1-win_rates[i]/100) * avg_losses[i])
                ax2.text(i, max(avg_wins[i], avg_losses[i]) + 0.3,
                       f'PF: {profit_factor:.2f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Model', fontsize=14)
        ax2.set_ylabel('Average Return (%)', fontsize=14)
        ax2.set_title('Win/Loss Magnitude', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Win Rate Analysis{title_suffix}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

def plot_metrics_table(metrics_dict: Dict[str, Dict],
                     metrics_to_show: List[tuple] = None,
                     title_suffix: str = "",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Enhanced formatted table of metrics with color coding
    
    Args:
        metrics_dict: Dictionary of metrics for each model
        metrics_to_show: List of (key, label, format, multiplier) tuples
        title_suffix: Additional text for title
        save_path: Path to save figure
    """
    set_publication_style('clean')
    
    # Default metrics to show
    if metrics_to_show is None:
        metrics_to_show = [
            ('total_return', 'Total Return', '{:.1f}%', 100),
            ('cagr', 'CAGR', '{:.1f}%', 100),
            ('annual_volatility', 'Volatility', '{:.1f}%', 100),
            ('sharpe_ratio', 'Sharpe', '{:.2f}', 1),
            ('sortino_ratio', 'Sortino', '{:.2f}', 1),
            ('calmar_ratio', 'Calmar', '{:.2f}', 1),
            ('max_drawdown', 'Max DD', '{:.1f}%', 100),
            ('positive_days', 'Win Rate', '{:.1f}%', 100)
        ]
    
    # Create DataFrame for easier manipulation
    data = []
    for name, metrics in metrics_dict.items():
        row = {'Model': name}
        for metric_key, label, format_str, multiplier in metrics_to_show:
            value = metrics.get(metric_key, 0) * multiplier
            row[label] = value
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4 + len(df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format values for display
    display_data = []
    for _, row in df.iterrows():
        display_row = [row['Model']]
        for metric_key, label, format_str, multiplier in metrics_to_show:
            if label in row:
                display_row.append(format_str.format(row[label]))
            else:
                display_row.append('N/A')
        display_data.append(display_row)
    
    # Create table
    columns = ['Model'] + [m[1] for m in metrics_to_show]
    table = ax.table(cellText=display_data,
                    colLabels=columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12] * len(columns))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Header styling
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code cells based on performance
    for i in range(1, len(display_data) + 1):
        model_name = display_data[i-1][0]
        
        # Model name cell
        table[(i, 0)].set_facecolor(MODEL_COLORS.get(model_name, '#FFFFFF'))
        table[(i, 0)].set_alpha(0.3)
        table[(i, 0)].set_text_props(weight='bold')
        
        # Color code performance metrics
        for j, (metric_key, _, _, _) in enumerate(metrics_to_show, 1):
            cell = table[(i, j)]
            
            # Get the actual value for comparison
            if j <= len(columns) - 1:
                value = df.iloc[i-1][columns[j]]
                
                # Color based on metric type and value
                if metric_key in ['total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']:
                    # Higher is better
                    if value > df[columns[j]].median():
                        cell.set_facecolor('#E8F5E9')  # Light green
                    elif value < df[columns[j]].median():
                        cell.set_facecolor('#FFEBEE')  # Light red
                elif metric_key in ['max_drawdown', 'annual_volatility']:
                    # Lower is better (but max_drawdown is negative)
                    if metric_key == 'max_drawdown':
                        if value > df[columns[j]].median():  # Less negative is better
                            cell.set_facecolor('#E8F5E9')
                        else:
                            cell.set_facecolor('#FFEBEE')
                    else:
                        if value < df[columns[j]].median():
                            cell.set_facecolor('#E8F5E9')
                        else:
                            cell.set_facecolor('#FFEBEE')
    
    # Add title
    ax.set_title(f'Performance Metrics Summary{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        ensure_dir_exists(save_path)
        save_figure(fig, save_path)
    
    return fig

# Alias functions for backward compatibility
plot_sharpe_bars = plot_sharpe_comparison
plot_returns_bars = plot_returns_comparison
plot_drawdown_bars = plot_drawdown_comparison