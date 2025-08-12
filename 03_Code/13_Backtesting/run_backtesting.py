"""
Main Backtesting Script for All LSTM Models
Runs comprehensive backtesting for daily/weekly, base/sentiment/meta models
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from a_lstm_strategy import (
    LSTMBinaryStrategy, LSTMProbabilityStrategy, 
    LSTMConfidenceStrategy, MetaModelStrategy
)
from b_backtesting_engine import BacktestEngine, BacktestAnalyzer, simulate_nav_simple
from c_evaluation_metrics import (
    calculate_comprehensive_metrics, statistical_tests,
    create_performance_summary
)
from d_visualization import (
    plot_cumulative_returns, plot_individual_analysis,
    plot_model_comparison, create_summary_latex_table
)


# Configuration
MODELS_DIR = '../../04_Models/'
PLOTS_DIR = '../../Plots/backtesting/'
RESULTS_DIR = './results/'
DATA_DIR = '../../02_Data_Processed/'

# Markets and frequencies
MARKETS = ['GDEA', 'HBEA']
FREQUENCIES = ['daily', 'weekly']
MODEL_TYPES = ['base', 'sentiment']

# Backtesting parameters
INITIAL_CAPITAL = 1000000
TRANSACTION_COST = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005         # 0.05% market impact


def load_price_data(market: str, frequency: str) -> pd.Series:
    """Load price data for backtesting"""
    if frequency == 'daily':
        path = f"{DATA_DIR}01_Carbon_Markets/01_Regional/{market}_forward_filled.parquet"
    else:  # weekly
        path = f"{DATA_DIR}11_Weekly_Aggregated/{market}_weekly.parquet"
    
    df = pd.read_parquet(path)
    return df['close']


def load_benchmark_strategy(prices: pd.Series) -> pd.Series:
    """Create Buy&Hold benchmark"""
    # Simple buy and hold - always long
    signals = pd.Series(1, index=prices.index)
    nav = simulate_nav_simple(prices, signals, INITIAL_CAPITAL)
    return nav


def run_model_backtest(market: str, frequency: str, model_type: str, 
                       with_costs: bool = True) -> dict:
    """
    Run backtest for a specific model
    
    Args:
        market: GDEA or HBEA
        frequency: daily or weekly
        model_type: base, sentiment, or meta
        with_costs: Include transaction costs
        
    Returns:
        results: Dictionary with NAV, metrics, and statistics
    """
    print(f"\n{'='*60}")
    print(f"Backtesting {frequency} {market} {model_type}")
    print(f"{'='*60}")
    
    # Load price data
    prices = load_price_data(market, frequency)
    print(f"Loaded {len(prices)} price points")
    
    # Determine model directory
    if model_type == 'meta':
        model_dir = f"{MODELS_DIR}meta_{frequency}_{market}"
    else:
        model_dir = f"{MODELS_DIR}{frequency}_{market}_{model_type}"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return None
    
    # Initialize strategy
    try:
        if model_type == 'meta':
            strategy = MetaModelStrategy(model_dir, market)
        else:
            # Try different strategy types
            strategy = LSTMBinaryStrategy(model_dir, market, model_type)
        
        # Generate signals
        signals = strategy.get_signals_with_dates()
        print(f"Generated {len(signals)} signals")
        
        # Skip if no signals
        if len(signals) == 0:
            print(f"No signals available for {model_type} model, skipping...")
            return None
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return None
    
    # Align signals with prices
    aligned_prices = prices[signals.index]
    
    # Initialize backtesting engine
    if with_costs:
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE
        )
        nav, stats = engine.simulate_nav(aligned_prices, signals)
    else:
        nav = simulate_nav_simple(aligned_prices, signals, INITIAL_CAPITAL)
        stats = {'total_trades': 0, 'total_costs': 0}
    
    print(f"Final NAV: {nav.iloc[-1]:.0f}")
    print(f"Total trades: {stats['total_trades']}")
    print(f"Transaction costs: {stats['total_costs']:.0f}")
    
    # Calculate metrics
    benchmark_nav = load_benchmark_strategy(aligned_prices)
    metrics = calculate_comprehensive_metrics(nav, benchmark_nav)
    
    # Statistical tests
    test_results = statistical_tests(nav, benchmark_nav)
    
    # Print key metrics
    print(f"\nKey Metrics:")
    print(f"  Total Return: {metrics['total_return']*100:.1f}%")
    print(f"  CAGR: {metrics['cagr']*100:.1f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"  Win Rate: {metrics['positive_days']*100:.1f}%")
    
    return {
        'nav': nav,
        'benchmark_nav': benchmark_nav,
        'metrics': metrics,
        'stats': stats,
        'test_results': test_results,
        'signals': signals
    }


def run_all_models():
    """Run backtesting for all model combinations"""
    
    all_results = {}
    
    for frequency in FREQUENCIES:
        for market in MARKETS:
            
            # Store results for comparison
            market_results = {}
            nav_dict = {}
            metrics_dict = {}
            
            # 1. Benchmark (Buy&Hold)
            prices = load_price_data(market, frequency)
            benchmark_nav = load_benchmark_strategy(prices)
            benchmark_metrics = calculate_comprehensive_metrics(benchmark_nav)
            
            nav_dict['Buy&Hold'] = benchmark_nav
            metrics_dict['Buy&Hold'] = benchmark_metrics
            
            # 2. Base model
            base_results = run_model_backtest(market, frequency, 'base')
            if base_results:
                nav_dict['Base'] = base_results['nav']
                metrics_dict['Base'] = base_results['metrics']
                market_results['base'] = base_results
            
            # 3. Sentiment model
            sent_results = run_model_backtest(market, frequency, 'sentiment')
            if sent_results:
                nav_dict['Sentiment'] = sent_results['nav']
                metrics_dict['Sentiment'] = sent_results['metrics']
                market_results['sentiment'] = sent_results
            
            # 4. Meta model
            meta_results = run_model_backtest(market, frequency, 'meta')
            if meta_results:
                nav_dict['Meta'] = meta_results['nav']
                metrics_dict['Meta'] = meta_results['metrics']
                market_results['meta'] = meta_results
            
            # Create comparison plots
            if nav_dict:
                # Cumulative returns comparison
                save_path = f"{PLOTS_DIR}{frequency}/{market}/cumulative_returns.png"
                plot_cumulative_returns(nav_dict, 
                                       title=f"{market} {frequency.capitalize()} Models",
                                       save_path=save_path)
                
                # Model comparison
                save_path = f"{PLOTS_DIR}comparison/{frequency}_{market}_comparison.png"
                plot_model_comparison(nav_dict, metrics_dict, market, frequency,
                                    save_path=save_path)
                
                # Individual model analysis
                for model_name, nav in nav_dict.items():
                    if model_name != 'Buy&Hold':
                        save_dir = f"{PLOTS_DIR}individual/{frequency}_{market}_{model_name.lower()}/"
                        plot_individual_analysis(nav, model_name, market, frequency,
                                               benchmark_nav=nav_dict.get('Buy&Hold'),
                                               save_dir=save_dir)
            
            # Store results
            all_results[f"{frequency}_{market}"] = {
                'nav_dict': nav_dict,
                'metrics_dict': metrics_dict,
                'results': market_results
            }
    
    return all_results


def create_summary_tables(all_results: dict):
    """Create summary tables and LaTeX output"""
    
    # Combine all metrics
    summary_data = []
    
    for key, data in all_results.items():
        frequency, market = key.split('_')
        
        for model_name, metrics in data['metrics_dict'].items():
            row = {
                'Frequency': frequency,
                'Market': market,
                'Model': model_name,
                'Total Return (%)': metrics['total_return'] * 100,
                'CAGR (%)': metrics['cagr'] * 100,
                'Volatility (%)': metrics['annual_volatility'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Win Rate (%)': metrics['positive_days'] * 100
            }
            summary_data.append(row)
    
    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = f"{RESULTS_DIR}metrics/all_models_summary.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")
    
    # Create LaTeX tables for each frequency
    for frequency in FREQUENCIES:
        freq_data = df_summary[df_summary['Frequency'] == frequency]
        
        # Pivot for better presentation
        pivot = freq_data.pivot_table(
            index=['Market', 'Model'],
            values=['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
            aggfunc='first'
        )
        
        # Save LaTeX
        latex_path = f"{PLOTS_DIR}paper_figures/table_{frequency}_performance.tex"
        os.makedirs(os.path.dirname(latex_path), exist_ok=True)
        
        with open(latex_path, 'w') as f:
            f.write(pivot.to_latex(float_format='%.2f'))
        
        print(f"Saved LaTeX table to {latex_path}")
    
    return df_summary


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Run LSTM backtesting')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'all'], default='all',
                       help='Market to backtest')
    parser.add_argument('--frequency', choices=['daily', 'weekly', 'all'], default='all',
                       help='Frequency to backtest')
    parser.add_argument('--model', choices=['base', 'sentiment', 'meta', 'all'], default='all',
                       help='Model type to backtest')
    parser.add_argument('--no-costs', action='store_true',
                       help='Run without transaction costs')
    
    args = parser.parse_args()
    
    print("="*80)
    print("LSTM MODEL BACKTESTING SYSTEM")
    print(f"Started at: {datetime.now()}")
    print("="*80)
    
    if args.market == 'all' and args.frequency == 'all' and args.model == 'all':
        # Run all combinations
        all_results = run_all_models()
        
        # Create summary tables
        df_summary = create_summary_tables(all_results)
        
        # Print final summary
        print("\n" + "="*80)
        print("BACKTESTING COMPLETE - SUMMARY")
        print("="*80)
        
        # Group by frequency and market
        for frequency in FREQUENCIES:
            for market in MARKETS:
                print(f"\n{frequency.upper()} {market}:")
                subset = df_summary[(df_summary['Frequency'] == frequency) & 
                                  (df_summary['Market'] == market)]
                
                for _, row in subset.iterrows():
                    print(f"  {row['Model']:15s}: Return={row['Total Return (%)']:6.1f}%, "
                          f"Sharpe={row['Sharpe Ratio']:5.2f}, "
                          f"MaxDD={row['Max Drawdown (%)']:6.1f}%")
    
    else:
        # Run specific combination
        markets = MARKETS if args.market == 'all' else [args.market]
        frequencies = FREQUENCIES if args.frequency == 'all' else [args.frequency]
        models = MODEL_TYPES + ['meta'] if args.model == 'all' else [args.model]
        
        for market in markets:
            for frequency in frequencies:
                for model in models:
                    results = run_model_backtest(market, frequency, model, 
                                               with_costs=not args.no_costs)
                    
                    if results:
                        # Save individual results
                        result_path = f"{RESULTS_DIR}metrics/{frequency}_{market}_{model}.json"
                        os.makedirs(os.path.dirname(result_path), exist_ok=True)
                        
                        # Convert to serializable format
                        save_metrics = {k: float(v) if isinstance(v, np.number) else v 
                                      for k, v in results['metrics'].items()}
                        
                        with open(result_path, 'w') as f:
                            json.dump(save_metrics, f, indent=2)
    
    print(f"\nCompleted at: {datetime.now()}")
    print("All results saved to:", RESULTS_DIR)
    print("All plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()