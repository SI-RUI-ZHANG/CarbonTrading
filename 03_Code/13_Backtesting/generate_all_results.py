"""
Generate comprehensive backtesting results and visualizations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from run_backtesting import run_all_models, create_summary_tables

def main():
    """Main execution"""
    print("="*80)
    print("COMPREHENSIVE BACKTESTING ANALYSIS")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    # Run all models (excluding meta for now)
    print("\n1. Running backtesting for all models...")
    all_results = {}
    
    # Markets and frequencies
    markets = ['GDEA', 'HBEA']
    frequencies = ['daily', 'weekly']
    model_types = ['base', 'sentiment']
    
    for frequency in frequencies:
        for market in markets:
            print(f"\n--- {frequency.upper()} {market} ---")
            
            # Import functions
            from run_backtesting import load_price_data, load_benchmark_strategy, run_model_backtest
            from c_evaluation_metrics import calculate_comprehensive_metrics
            from d_visualization import (plot_individual_analysis, plot_cumulative_returns, 
                                         plot_sharpe_comparison, plot_risk_return_scatter,
                                         create_summary_latex_table)
            
            # Store results
            market_results = {}
            nav_dict = {}
            metrics_dict = {}
            
            # 1. Benchmark (Buy&Hold)
            prices = load_price_data(market, frequency)
            benchmark_nav = load_benchmark_strategy(prices)
            benchmark_metrics = calculate_comprehensive_metrics(benchmark_nav)
            
            nav_dict['Buy&Hold'] = benchmark_nav
            metrics_dict['Buy&Hold'] = benchmark_metrics
            print(f"Buy&Hold: Return={benchmark_metrics['total_return']*100:.1f}%, Sharpe={benchmark_metrics['sharpe_ratio']:.2f}")
            
            # 2. Models
            for model_type in model_types:
                results = run_model_backtest(market, frequency, model_type)
                if results:
                    nav_dict[model_type.capitalize()] = results['nav']
                    metrics_dict[model_type.capitalize()] = results['metrics']
                    market_results[model_type] = results
                    print(f"{model_type.capitalize()}: Return={results['metrics']['total_return']*100:.1f}%, Sharpe={results['metrics']['sharpe_ratio']:.2f}")
            
            # Create visualizations
            print(f"\nGenerating visualizations for {frequency} {market}...")
            
            # 1. Individual model analysis (detailed plots for each model)
            for model_name, nav in nav_dict.items():
                save_dir = f"../../Plots/backtesting/individual/{frequency}_{market}_{model_name.lower()}/"
                plot_individual_analysis(nav, model_name, market, frequency,
                                       benchmark_nav if model_name != 'Buy&Hold' else None,
                                       save_dir=save_dir)
                print(f"  Generated individual analysis for {model_name}")
            
            # 2. Simple comparisons (max 3 models for clarity)
            # Cumulative returns comparison
            save_path = f"../../Plots/backtesting/comparisons/{frequency}_{market}_cumulative.png"
            plot_cumulative_returns(nav_dict, 
                                  title=f"{market} {frequency.capitalize()} - Cumulative Returns",
                                  save_path=save_path)
            print(f"  Saved: cumulative returns comparison")
            
            # Sharpe ratio comparison
            save_path = f"../../Plots/backtesting/comparisons/{frequency}_{market}_sharpe.png"
            plot_sharpe_comparison(metrics_dict, market, frequency, save_path=save_path)
            print(f"  Saved: Sharpe ratio comparison")
            
            # Risk-return scatter
            save_path = f"../../Plots/backtesting/comparisons/{frequency}_{market}_risk_return.png"
            plot_risk_return_scatter(metrics_dict, market, frequency, save_path=save_path)
            print(f"  Saved: risk-return scatter")
            
            # 3. Summary table (LaTeX format for paper)
            save_path = f"../../Plots/backtesting/summary/{frequency}_{market}_table.tex"
            create_summary_latex_table(metrics_dict, market, frequency, save_path=save_path)
            print(f"  Saved: LaTeX summary table")
            
            # Store results
            all_results[f"{frequency}_{market}"] = {
                'nav_dict': nav_dict,
                'metrics_dict': metrics_dict,
                'results': market_results
            }
    
    # Create summary tables
    print("\n2. Creating summary tables...")
    df_summary = create_summary_tables(all_results)
    
    # Print consolidated results
    print("\n" + "="*80)
    print("CONSOLIDATED RESULTS")
    print("="*80)
    
    # Create performance comparison table
    comparison = []
    for key, data in all_results.items():
        frequency, market = key.split('_')
        for model_name, metrics in data['metrics_dict'].items():
            comparison.append({
                'Frequency': frequency,
                'Market': market,
                'Model': model_name,
                'Total Return (%)': f"{metrics['total_return']*100:.1f}",
                'CAGR (%)': f"{metrics['cagr']*100:.1f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD (%)': f"{metrics['max_drawdown']*100:.1f}",
                'Win Rate (%)': f"{metrics.get('positive_days', 0)*100:.1f}"
            })
    
    df_comparison = pd.DataFrame(comparison)
    
    # Print by frequency
    for frequency in frequencies:
        print(f"\n{frequency.upper()} MODELS:")
        freq_data = df_comparison[df_comparison['Frequency'] == frequency]
        
        # Pivot for better display
        pivot = freq_data.pivot_table(
            index='Model',
            columns='Market',
            values='Sharpe Ratio',
            aggfunc='first'
        )
        print(pivot)
    
    # Save final summary
    summary_path = "../../Plots/backtesting/performance_summary.csv"
    df_comparison.to_csv(summary_path, index=False)
    print(f"\nSaved final summary to: {summary_path}")
    
    print(f"\nCompleted at: {datetime.now()}")
    print("\nPlots organized by type:")
    print("  Individual model analysis: ../../Plots/backtesting/individual/")
    print("  Model comparisons: ../../Plots/backtesting/comparisons/")
    print("  Summary tables: ../../Plots/backtesting/summary/")
    print("  Performance summary CSV: ../../Plots/backtesting/performance_summary.csv")

if __name__ == "__main__":
    main()