"""
Configuration and styling for all visualization modules
Ensures consistency across all plots
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Color scheme - Enhanced for better visibility
MODEL_COLORS = {
    'Buy&Hold': '#808080',  # Gray
    'Base': '#1f77b4',      # Blue
    'Sentiment': '#2ca02c', # Green
    'Meta': '#d62728',      # Red
    'Benchmark': '#808080'  # Gray (alias for Buy&Hold)
}

# Colorblind-friendly palette option
MODEL_COLORS_CB = {
    'Buy&Hold': '#999999',  # Gray
    'Base': '#0173B2',      # Blue
    'Sentiment': '#DE8F05', # Orange
    'Meta': '#CC78BC',      # Light purple
    'Benchmark': '#999999'  # Gray
}

# Diverging colormap for heatmaps (centered at zero)
DIVERGING_CMAP = 'RdYlGn'  # Red-Yellow-Green
DIVERGING_CMAP_R = 'RdYlGn_r'  # Reversed version

# Line styles
MODEL_LINESTYLES = {
    'Buy&Hold': '--',
    'Base': '-',
    'Sentiment': '-',
    'Meta': '-',
    'Benchmark': '--'
}

# Line widths
MODEL_LINEWIDTHS = {
    'Buy&Hold': 2.0,
    'Base': 2.5,
    'Sentiment': 2.5,
    'Meta': 2.5,
    'Benchmark': 2.0
}

# Alpha values for transparency
MODEL_ALPHAS = {
    'Buy&Hold': 0.7,
    'Base': 1.0,
    'Sentiment': 1.0,
    'Meta': 1.0,
    'Benchmark': 0.7
}

# Figure sizes
FIGURE_SIZES = {
    'single': (12, 7),        # Individual plots
    'comparison': (14, 8),    # Comparison plots
    'bar': (10, 6),          # Bar charts
    'scatter': (10, 8),      # Scatter plots
    'heatmap': (12, 8),      # Heatmap plots
    'multi_panel': (16, 10)  # Multi-panel plots
}

# Base directory for plots
PLOTS_BASE_DIR = '../../Plots/backtesting/'

def set_publication_style(style='clean'):
    """
    Set matplotlib parameters for publication-quality figures
    
    Args:
        style: 'clean', 'grid', or 'academic'
    """
    # Base settings
    base_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'png',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    }
    
    # Style-specific settings
    if style == 'clean':
        base_params.update({
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True
        })
    elif style == 'grid':
        base_params.update({
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#CCCCCC',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True
        })
    elif style == 'academic':
        base_params.update({
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.color': '#DDDDDD',
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'mathtext.fontset': 'dejavuserif'
        })
    
    plt.rcParams.update(base_params)

def format_axis_dates(ax, interval='month', rotation=45):
    """
    Format date axis for better readability
    
    Args:
        ax: Matplotlib axis
        interval: 'day', 'week', 'month', 'quarter', 'year'
        rotation: Rotation angle for labels
    """
    import matplotlib.dates as mdates
    
    if interval == 'day':
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif interval == 'week':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif interval == 'month':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif interval == 'quarter':
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-Q%q'))
    elif interval == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')

def add_value_labels(ax, bars, format_str='{:.1f}', offset=0.01):
    """
    Add value labels on top of bars
    
    Args:
        ax: Matplotlib axis
        bars: Bar container
        format_str: Format string for values
        offset: Vertical offset for labels
    """
    for bar in bars:
        height = bar.get_height()
        if height != 0:  # Only label non-zero bars
            ax.annotate(format_str.format(height),
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),  # Offset
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=10)

def add_stat_annotation(ax, x, y, text, boxstyle="round,pad=0.3"):
    """
    Add statistical annotation to plot
    
    Args:
        ax: Matplotlib axis
        x, y: Position in data coordinates
        text: Annotation text
        boxstyle: Box style for annotation
    """
    ax.annotate(text,
                xy=(x, y),
                xycoords='data',
                bbox=dict(boxstyle=boxstyle, facecolor='white', 
                         edgecolor='gray', alpha=0.8),
                fontsize=10,
                ha='left')

def create_legend(ax, loc='best', ncol=1, title=None):
    """
    Create a formatted legend
    
    Args:
        ax: Matplotlib axis
        loc: Legend location
        ncol: Number of columns
        title: Legend title
    """
    legend = ax.legend(loc=loc, ncol=ncol, title=title,
                      frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, edgecolor='#CCCCCC')
    if title:
        legend.get_title().set_fontsize(11)
        legend.get_title().set_fontweight('bold')
    return legend

def save_figure(fig, save_path, dpi=300, tight=True):
    """
    Save figure with consistent settings
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save
        dpi: Resolution
        tight: Use tight layout
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if tight:
        fig.tight_layout()
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return save_path

def add_period_shading(ax, periods, alpha=0.2, color='gray'):
    """
    Add shaded regions to highlight specific periods
    
    Args:
        ax: Matplotlib axis
        periods: List of (start, end, label) tuples
        alpha: Transparency
        color: Shading color
    """
    for start, end, label in periods:
        ax.axvspan(start, end, alpha=alpha, color=color, label=label)

def smart_label_placement(ax, points, labels, min_distance=0.05):
    """
    Place labels to avoid overlaps
    
    Args:
        ax: Matplotlib axis
        points: List of (x, y) coordinates
        labels: List of label strings
        min_distance: Minimum distance between labels
    """
    from adjustText import adjust_text
    texts = []
    for (x, y), label in zip(points, labels):
        texts.append(ax.annotate(label, (x, y), fontsize=10))
    
    try:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    except ImportError:
        # Fallback if adjust_text not available
        pass
    
    return texts

def add_smooth_trend(ax, x, y, window=30, label='Trend', color='red', alpha=0.7):
    """
    Add smoothed trend line to noisy data
    
    Args:
        ax: Matplotlib axis
        x: X data (dates)
        y: Y data (values)
        window: Smoothing window
        label: Label for legend
        color: Line color
        alpha: Transparency
    """
    import pandas as pd
    # Convert to series for rolling mean
    series = pd.Series(y, index=x)
    smoothed = series.rolling(window=window, center=True).mean()
    ax.plot(smoothed.index, smoothed.values, color=color, alpha=alpha, 
            linewidth=2, label=label, linestyle='--')
    return smoothed