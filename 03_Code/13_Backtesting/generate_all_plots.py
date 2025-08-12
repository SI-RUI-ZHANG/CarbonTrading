"""
Master script to generate all visualization plots
Uses the new modular visualization system
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import all visualization modules
from plot_config import PLOTS_BASE_DIR
from individual_plots import (
    plot_cumulative_returns as plot_individual_returns,
    plot_drawdown as plot_individual_drawdown,
    plot_rolling_sharpe as plot_individual_sharpe,
    plot_monthly_returns_heatmap,
    plot_return_distribution,
    plot_rolling_volatility as plot_individual_volatility
)
from comparison_plots import (
    plot_cumulative_returns_comparison,
    plot_drawdown_comparison,
    plot_rolling_sharpe_comparison,
    plot_rolling_volatility_comparison,
    plot_performance_ribbon
)
from metric_plots import (
    plot_sharpe_bars,
    plot_returns_bars,
    plot_drawdown_bars,
    plot_risk_return_scatter,
    plot_metrics_table,
    plot_win_rate_comparison
)

# Import backtesting functions
from run_backtesting import (
    load_price_data,
    load_benchmark_strategy,
    run_model_backtest
)
from c_evaluation_metrics import calculate_comprehensive_metrics

def generate_individual_plots(nav: pd.Series, 
                            metrics: dict,
                            model_name: str,
                            frequency: str,
                            market: str,
                            benchmark_nav: pd.Series = None):
    """Generate all individual plots for a single model"""
    
    base_path = f"{PLOTS_BASE_DIR}individual/{frequency}/{market}/{model_name.lower()}/"
    os.makedirs(base_path, exist_ok=True)
    
    print(f"  Generating individual plots for {model_name}...")
    
    # 1. Cumulative returns
    plot_individual_returns(
        nav, model_name, benchmark_nav,
        save_path=f"{base_path}cumulative_returns.png"
    )
    
    # 2. Drawdown
    plot_individual_drawdown(
        nav, model_name,
        save_path=f"{base_path}drawdown.png"
    )
    
    # 3. Rolling Sharpe
    window = 252 if frequency == 'daily' else 52
    plot_individual_sharpe(
        nav, model_name, window=window,
        save_path=f"{base_path}rolling_sharpe.png"
    )
    
    # 4. Monthly returns heatmap
    plot_monthly_returns_heatmap(
        nav, model_name,
        save_path=f"{base_path}monthly_heatmap.png"
    )
    
    # 5. Return distribution
    plot_return_distribution(
        nav, model_name,
        save_path=f"{base_path}return_distribution.png"
    )
    
    # 6. Rolling volatility
    plot_individual_volatility(
        nav, model_name,
        windows=[30, 60, 120] if frequency == 'daily' else [4, 8, 12],
        save_path=f"{base_path}rolling_volatility.png"
    )

def generate_comparison_plots(nav_dict: dict,
                             metrics_dict: dict,
                             frequency: str,
                             market: str):
    """Generate comparison plots for multiple models"""
    
    base_path = f"{PLOTS_BASE_DIR}comparisons/{frequency}/{market}/"
    os.makedirs(base_path, exist_ok=True)
    
    print(f"  Generating comparison plots...")
    
    title_suffix = f" - {market} {frequency.capitalize()}"
    
    # 1. Aligned cumulative returns
    plot_cumulative_returns_comparison(
        nav_dict, title_suffix=title_suffix, align_start=True,
        save_path=f"{base_path}cumulative_returns_aligned.png"
    )
    
    # 2. Drawdown comparison
    plot_drawdown_comparison(
        nav_dict, title_suffix=title_suffix,
        save_path=f"{base_path}drawdown_comparison.png"
    )
    
    # 3. Rolling Sharpe comparison
    window = 252 if frequency == 'daily' else 52
    plot_rolling_sharpe_comparison(
        nav_dict, window=window, title_suffix=title_suffix,
        save_path=f"{base_path}rolling_sharpe_comparison.png"
    )
    
    # 4. Rolling volatility comparison
    window = 60 if frequency == 'daily' else 8
    plot_rolling_volatility_comparison(
        nav_dict, window=window, title_suffix=title_suffix,
        save_path=f"{base_path}rolling_volatility_comparison.png"
    )
    
    # 5. Performance ribbon
    plot_performance_ribbon(
        nav_dict, metric='sharpe', window=window*4, title_suffix=title_suffix,
        save_path=f"{base_path}performance_ribbon.png"
    )

def generate_metric_plots(metrics_dict: dict,
                         frequency: str,
                         market: str):
    """Generate metric bar charts and scatter plots"""
    
    base_path = f"{PLOTS_BASE_DIR}metrics/{frequency}/{market}/"
    os.makedirs(base_path, exist_ok=True)
    
    print(f"  Generating metric plots...")
    
    title_suffix = f" - {market} {frequency.capitalize()}"
    
    # 1. Sharpe ratio bars
    plot_sharpe_bars(
        metrics_dict, title_suffix=title_suffix,
        save_path=f"{base_path}sharpe_bars.png"
    )
    
    # 2. Total returns bars
    plot_returns_bars(
        metrics_dict, return_type='total', title_suffix=title_suffix,
        save_path=f"{base_path}total_returns_bars.png"
    )
    
    # 3. CAGR bars
    plot_returns_bars(
        metrics_dict, return_type='cagr', title_suffix=title_suffix,
        save_path=f"{base_path}cagr_bars.png"
    )
    
    # 4. Drawdown bars
    plot_drawdown_bars(
        metrics_dict, title_suffix=title_suffix,
        save_path=f"{base_path}drawdown_bars.png"
    )
    
    # 5. Risk-return scatter
    plot_risk_return_scatter(
        metrics_dict, title_suffix=title_suffix,
        save_path=f"{base_path}risk_return_scatter.png"
    )
    
    # 6. Metrics table
    plot_metrics_table(
        metrics_dict, title_suffix=title_suffix,
        save_path=f"{base_path}metrics_table.png"
    )
    
    # 7. Win rate comparison
    plot_win_rate_comparison(
        metrics_dict, title_suffix=title_suffix,
        save_path=f"{base_path}win_rate_comparison.png"
    )

def main():
    """Main execution function"""
    
    print("="*80)
    print("GENERATING ALL VISUALIZATION PLOTS")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    # Configuration
    markets = ['GDEA', 'HBEA']
    frequencies = ['daily', 'weekly']
    model_types = ['base', 'sentiment']
    
    # Summary for paper-ready plots
    paper_ready_plots = []
    
    for frequency in frequencies:
        for market in markets:
            print(f"\n{'='*60}")
            print(f"Processing {frequency.upper()} {market}")
            print(f"{'='*60}")
            
            # Load benchmark
            prices = load_price_data(market, frequency)
            benchmark_nav = load_benchmark_strategy(prices)
            benchmark_metrics = calculate_comprehensive_metrics(benchmark_nav)
            
            # Store results
            nav_dict = {'Buy&Hold': benchmark_nav}
            metrics_dict = {'Buy&Hold': benchmark_metrics}
            
            # Process each model
            for model_type in model_types:
                print(f"\nProcessing {model_type} model...")
                results = run_model_backtest(market, frequency, model_type)
                
                if results:
                    model_name = model_type.capitalize()
                    nav_dict[model_name] = results['nav']
                    metrics_dict[model_name] = results['metrics']
                    
                    # Generate individual plots
                    generate_individual_plots(
                        results['nav'], 
                        results['metrics'],
                        model_name,
                        frequency,
                        market,
                        benchmark_nav
                    )
                    
                    print(f"  ✓ Individual plots completed for {model_name}")
            
            # Generate comparison plots
            print(f"\nGenerating comparison plots for {frequency} {market}...")
            generate_comparison_plots(nav_dict, metrics_dict, frequency, market)
            print("  ✓ Comparison plots completed")
            
            # Generate metric plots
            print(f"\nGenerating metric plots for {frequency} {market}...")
            generate_metric_plots(metrics_dict, frequency, market)
            print("  ✓ Metric plots completed")
            
            # Track best plots for paper
            best_sharpe_model = max(metrics_dict.items(), 
                                  key=lambda x: x[1].get('sharpe_ratio', 0))
            paper_ready_plots.append({
                'frequency': frequency,
                'market': market,
                'best_model': best_sharpe_model[0],
                'sharpe': best_sharpe_model[1]['sharpe_ratio']
            })
    
    # Copy best plots to paper_ready folder
    print("\n" + "="*80)
    print("SELECTING PAPER-READY PLOTS")
    print("="*80)
    
    paper_path = f"{PLOTS_BASE_DIR}paper_ready/"
    os.makedirs(paper_path, exist_ok=True)
    
    # Select key plots for paper
    import shutil
    
    # Best performing model plots
    for item in paper_ready_plots:
        freq = item['frequency']
        mkt = item['market']
        model = item['best_model'].lower()
        
        # Copy key individual plots
        src = f"{PLOTS_BASE_DIR}individual/{freq}/{mkt}/{model}/cumulative_returns.png"
        if os.path.exists(src):
            dst = f"{paper_path}{freq}_{mkt}_{model}_returns.png"
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {freq} {mkt} {model} returns")
    
    # Copy all comparison plots
    for freq in frequencies:
        for mkt in markets:
            src = f"{PLOTS_BASE_DIR}comparisons/{freq}/{mkt}/cumulative_returns_aligned.png"
            if os.path.exists(src):
                dst = f"{paper_path}{freq}_{mkt}_comparison.png"
                shutil.copy2(src, dst)
                print(f"  ✓ Copied {freq} {mkt} comparison")
            
            # Risk-return scatter
            src = f"{PLOTS_BASE_DIR}metrics/{freq}/{mkt}/risk_return_scatter.png"
            if os.path.exists(src):
                dst = f"{paper_path}{freq}_{mkt}_risk_return.png"
                shutil.copy2(src, dst)
                print(f"  ✓ Copied {freq} {mkt} risk-return")
    
    print(f"\nCompleted at: {datetime.now()}")
    print(f"All plots saved to: {PLOTS_BASE_DIR}")
    print(f"Paper-ready plots in: {paper_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY OF BEST MODELS")
    print("="*80)
    
    for item in paper_ready_plots:
        print(f"{item['frequency'].upper():6} {item['market']:4} - "
              f"Best: {item['best_model']:10} "
              f"(Sharpe: {item['sharpe']:.2f})")

if __name__ == "__main__":
    main()