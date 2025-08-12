"""
Comprehensive Performance Metrics for Backtesting
Includes financial metrics, risk measures, and statistical tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


TRADING_DAYS = 252  # Annual trading days


def calculate_comprehensive_metrics(nav_series: pd.Series,
                                   benchmark_nav: Optional[pd.Series] = None,
                                   risk_free_rate: float = 0.02,
                                   confidence_level: float = 0.95) -> Dict:
    """
    Calculate comprehensive performance metrics
    
    Args:
        nav_series: Portfolio NAV time series
        benchmark_nav: Benchmark NAV (e.g., Buy&Hold)
        risk_free_rate: Annual risk-free rate
        confidence_level: Confidence level for VaR/CVaR
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    metrics = {}
    
    # Basic return metrics
    returns = nav_series.pct_change().dropna()
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    
    # Time period
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25
    
    # Return metrics
    metrics['total_return'] = total_return
    metrics['cagr'] = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    metrics['mean_daily_return'] = returns.mean()
    metrics['annual_return'] = returns.mean() * TRADING_DAYS
    
    # Risk metrics
    metrics['daily_volatility'] = returns.std()
    metrics['annual_volatility'] = returns.std() * np.sqrt(TRADING_DAYS)
    metrics['downside_volatility'] = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    
    # Risk-adjusted returns
    daily_rf = risk_free_rate / TRADING_DAYS
    excess_returns = returns - daily_rf
    
    metrics['sharpe_ratio'] = (returns.mean() - daily_rf) / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() > 0 else 0
    
    # Sortino ratio (uses downside volatility)
    downside_returns = returns[returns < daily_rf]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std()
        metrics['sortino_ratio'] = (returns.mean() - daily_rf) / downside_std * np.sqrt(TRADING_DAYS) if downside_std > 0 else 0
    else:
        metrics['sortino_ratio'] = np.inf
    
    # Maximum drawdown
    running_max = nav_series.expanding().max()
    drawdown = (nav_series - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    # Calmar ratio (CAGR / Max Drawdown)
    metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
    
    # Value at Risk (VaR) and Conditional VaR (CVaR)
    var_percentile = (1 - confidence_level) * 100
    metrics['var_95'] = np.percentile(returns, var_percentile)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    
    # Higher moments
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # Benchmark comparison
    if benchmark_nav is not None:
        benchmark_returns = benchmark_nav.pct_change().dropna()
        
        # Align returns
        aligned = pd.DataFrame({
            'portfolio': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) > 1:
            # Beta and Alpha
            covariance = aligned.cov()
            variance_benchmark = covariance.loc['benchmark', 'benchmark']
            
            if variance_benchmark > 0:
                metrics['beta'] = covariance.loc['portfolio', 'benchmark'] / variance_benchmark
                metrics['alpha'] = metrics['annual_return'] - (risk_free_rate + metrics['beta'] * (benchmark_returns.mean() * TRADING_DAYS - risk_free_rate))
            
            # Information ratio
            active_returns = aligned['portfolio'] - aligned['benchmark']
            tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
            metrics['information_ratio'] = active_returns.mean() * TRADING_DAYS / tracking_error if tracking_error > 0 else 0
            
            # Win rate against benchmark
            metrics['win_rate_vs_benchmark'] = (aligned['portfolio'] > aligned['benchmark']).mean()
    
    # Additional metrics
    metrics['positive_days'] = (returns > 0).mean()
    metrics['negative_days'] = (returns < 0).mean()
    metrics['best_day'] = returns.max()
    metrics['worst_day'] = returns.min()
    
    # Recovery metrics
    drawdown_periods = identify_drawdown_periods(nav_series)
    if drawdown_periods:
        recovery_times = [p['recovery_days'] for p in drawdown_periods if p['recovery_days'] is not None]
        metrics['avg_recovery_days'] = np.mean(recovery_times) if recovery_times else None
        metrics['max_recovery_days'] = max(recovery_times) if recovery_times else None
    
    return metrics


def calculate_rolling_metrics(nav_series: pd.Series,
                             window: int = 252,
                             risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Calculate rolling performance metrics
    
    Args:
        nav_series: Portfolio NAV
        window: Rolling window size (default 252 days = 1 year)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        rolling_metrics: DataFrame with rolling metrics
    """
    returns = nav_series.pct_change().dropna()
    daily_rf = risk_free_rate / TRADING_DAYS
    
    rolling = pd.DataFrame(index=returns.index)
    
    # Rolling returns
    rolling['return'] = returns.rolling(window).mean() * TRADING_DAYS
    
    # Rolling volatility
    rolling['volatility'] = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    
    # Rolling Sharpe
    rolling['sharpe'] = (returns.rolling(window).mean() - daily_rf) / returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    
    # Rolling maximum drawdown
    rolling['max_dd'] = nav_series.rolling(window).apply(
        lambda x: (x - x.expanding().max()).min() / x.expanding().max().iloc[-1]
    )
    
    return rolling


def identify_drawdown_periods(nav_series: pd.Series) -> list:
    """
    Identify all drawdown periods
    
    Args:
        nav_series: NAV time series
        
    Returns:
        periods: List of drawdown period dictionaries
    """
    running_max = nav_series.expanding().max()
    drawdown = (nav_series - running_max) / running_max
    
    periods = []
    in_drawdown = False
    start_idx = None
    peak_value = None
    
    for i, (idx, dd) in enumerate(drawdown.items()):
        if dd < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = idx
            peak_value = running_max.iloc[i]
            
        elif dd >= 0 and in_drawdown:
            # End of drawdown (recovery)
            trough_idx = drawdown[start_idx:idx].idxmin()
            trough_value = nav_series[trough_idx]
            
            period = {
                'start': start_idx,
                'trough': trough_idx,
                'end': idx,
                'peak_value': peak_value,
                'trough_value': trough_value,
                'drawdown': drawdown[trough_idx],
                'duration_days': (idx - start_idx).days,
                'recovery_days': (idx - trough_idx).days
            }
            periods.append(period)
            in_drawdown = False
    
    # Handle ongoing drawdown
    if in_drawdown:
        trough_idx = drawdown[start_idx:].idxmin()
        period = {
            'start': start_idx,
            'trough': trough_idx,
            'end': None,
            'peak_value': peak_value,
            'trough_value': nav_series[trough_idx],
            'drawdown': drawdown[trough_idx],
            'duration_days': (nav_series.index[-1] - start_idx).days,
            'recovery_days': None  # Not recovered
        }
        periods.append(period)
    
    return periods


def calculate_monthly_returns(nav_series: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns for heatmap visualization
    
    Args:
        nav_series: NAV time series
        
    Returns:
        monthly_returns: DataFrame with years as rows, months as columns
    """
    # Resample to monthly
    monthly_nav = nav_series.resample('M').last()
    monthly_returns = monthly_nav.pct_change().dropna()
    
    # Create matrix
    returns_matrix = pd.DataFrame(index=sorted(monthly_returns.index.year.unique()),
                                 columns=range(1, 13))
    
    for date, ret in monthly_returns.items():
        returns_matrix.loc[date.year, date.month] = ret
    
    return returns_matrix


def statistical_tests(strategy_nav: pd.Series,
                     benchmark_nav: pd.Series,
                     confidence_level: float = 0.95) -> Dict:
    """
    Perform statistical significance tests
    
    Args:
        strategy_nav: Strategy NAV series
        benchmark_nav: Benchmark NAV series
        confidence_level: Confidence level for tests
        
    Returns:
        test_results: Dictionary of test results
    """
    strategy_returns = strategy_nav.pct_change().dropna()
    benchmark_returns = benchmark_nav.pct_change().dropna()
    
    # Align returns
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    results = {}
    
    # T-test for mean returns
    t_stat, p_value = stats.ttest_ind(aligned['strategy'], aligned['benchmark'])
    results['ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < (1 - confidence_level)
    }
    
    # Paired t-test (for same time periods)
    diff = aligned['strategy'] - aligned['benchmark']
    t_stat_paired, p_value_paired = stats.ttest_1samp(diff, 0)
    results['paired_ttest'] = {
        't_statistic': t_stat_paired,
        'p_value': p_value_paired,
        'significant': p_value_paired < (1 - confidence_level)
    }
    
    # Mann-Whitney U test (non-parametric)
    u_stat, p_value_mw = stats.mannwhitneyu(aligned['strategy'], aligned['benchmark'])
    results['mann_whitney'] = {
        'u_statistic': u_stat,
        'p_value': p_value_mw,
        'significant': p_value_mw < (1 - confidence_level)
    }
    
    # Sharpe ratio test (using bootstrap)
    sharpe_diff = bootstrap_sharpe_test(aligned['strategy'].values, 
                                       aligned['benchmark'].values,
                                       n_bootstrap=1000)
    results['sharpe_test'] = sharpe_diff
    
    return results


def bootstrap_sharpe_test(returns1: np.ndarray,
                         returns2: np.ndarray,
                         n_bootstrap: int = 1000,
                         confidence_level: float = 0.95) -> Dict:
    """
    Bootstrap test for Sharpe ratio difference
    
    Args:
        returns1: First strategy returns
        returns2: Second strategy returns
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        
    Returns:
        results: Test results
    """
    def sharpe(returns):
        return returns.mean() / returns.std() * np.sqrt(TRADING_DAYS) if returns.std() > 0 else 0
    
    sharpe1 = sharpe(returns1)
    sharpe2 = sharpe(returns2)
    observed_diff = sharpe1 - sharpe2
    
    # Bootstrap
    n = len(returns1)
    diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_sharpe1 = sharpe(returns1[idx])
        boot_sharpe2 = sharpe(returns2[idx])
        diffs.append(boot_sharpe1 - boot_sharpe2)
    
    diffs = np.array(diffs)
    
    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(diffs, alpha/2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha/2) * 100)
    
    # P-value
    p_value = np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)
    
    return {
        'sharpe1': sharpe1,
        'sharpe2': sharpe2,
        'difference': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value * 2,  # Two-tailed
        'significant': (ci_lower > 0) or (ci_upper < 0)
    }


def create_performance_summary(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a formatted performance summary table
    
    Args:
        metrics_dict: Dictionary of model names to metrics
        
    Returns:
        summary: Formatted DataFrame
    """
    rows = []
    
    for model_name, metrics in metrics_dict.items():
        row = {
            'Model': model_name,
            'Return (%)': f"{metrics['total_return']*100:.1f}",
            'CAGR (%)': f"{metrics['cagr']*100:.1f}",
            'Volatility (%)': f"{metrics['annual_volatility']*100:.1f}",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'Sortino': f"{metrics['sortino_ratio']:.2f}",
            'Calmar': f"{metrics['calmar_ratio']:.2f}",
            'Max DD (%)': f"{metrics['max_drawdown']*100:.1f}",
            'Win Rate (%)': f"{metrics.get('positive_days', 0)*100:.1f}"
        }
        rows.append(row)
    
    return pd.DataFrame(rows)