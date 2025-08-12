"""
Weekly LSTM Training with Walk-Forward Cross-Validation
Always uses walk-forward validation for more reliable evaluation
Handles both sentiment and non-sentiment versions based on config
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict

from config import Config
from walk_forward import WalkForwardValidator

# Import performance tracker
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.performance_tracker import update_performance_summary
except ImportError:
    # Fallback if utils is not a package
    sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
    from performance_tracker import update_performance_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_weekly_data(config) -> pd.DataFrame:
    """
    Load weekly aggregated data based on configuration
    
    Returns:
        DataFrame with weekly data
    """
    # Determine which file to load based on sentiment flag
    if config.USE_SENTIMENT:
        data_path = os.path.join(config.AGGREGATED_DATA_DIR, f'{config.MARKET}_weekly_with_sentiment.parquet')
    else:
        data_path = os.path.join(config.AGGREGATED_DATA_DIR, f'{config.MARKET}_weekly.parquet')
    
    logger.info(f"Loading weekly data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"Loaded {len(df)} weeks of data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df


def main():
    """Main execution - runs all combinations of markets and sentiment by default"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Weekly LSTM Model with Walk-Forward Cross-Validation')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--sentiment', choices=['base', 'sentiment', 'both'],
                       default='both', help='Feature set(s) to use (default: both)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Determine which sentiment options to run
    if args.sentiment == 'both':
        sentiment_options = [False, True]  # Base first, then with sentiment
    elif args.sentiment == 'base':
        sentiment_options = [False]
    else:  # sentiment
        sentiment_options = [True]
    
    # Store results for all combinations
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("WEEKLY LSTM CARBON PRICE DIRECTION PREDICTION")
    logger.info("Walk-Forward Cross-Validation")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Features: {args.sentiment}")
    logger.info("Validation: 14 walks, ~420 test samples per model")
    logger.info("="*80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model for each combination
    for market in markets:
        for use_sentiment in sentiment_options:
            config_name = f"{market}_{'sentiment' if use_sentiment else 'base'}"
            
            logger.info("\n" + "="*80)
            logger.info(f"PROCESSING: {config_name}")
            logger.info("="*80)
            
            try:
                # Create config for this combination
                config = Config(market=market, use_sentiment=use_sentiment)
                logger.info(f"Market: {config.MARKET}")
                logger.info(f"Use Sentiment: {config.USE_SENTIMENT}")
                logger.info(f"Output: {config.output_dir}")
                
                # Load full dataset
                df = load_weekly_data(config)
                
                # Create validator and run walk-forward
                logger.info("Starting walk-forward cross-validation...")
                validator = WalkForwardValidator(config)
                aggregated_metrics = validator.run_walk_forward(df)
                
                # Save all outputs
                validator.save_aggregated_outputs(config.OUTPUT_DIR)
                
                # Use aggregated metrics for summary
                all_metrics[config_name] = aggregated_metrics
                
                # Update performance summary with aggregated metrics
                update_performance_summary(
                    model_type=f'weekly_{"sentiment" if config.USE_SENTIMENT else "base"}',
                    market=config.MARKET,
                    metrics=aggregated_metrics,
                    features_type='+Sentiment' if config.USE_SENTIMENT else 'Base',
                    summary_path='../../04_Models/performance_summary.txt'
                )
                
                logger.info(f"✅ {config_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {config_name} failed: {str(e)}")
                import traceback
                traceback.print_exc()
                all_metrics[config_name] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    for config_name, metrics in all_metrics.items():
        if 'error' not in metrics:
            logger.info(f"{config_name}: Accuracy={metrics.get('test_accuracy', 0):.3f}, "
                       f"Precision={metrics.get('test_precision', 0):.3f}, "
                       f"Recall={metrics.get('test_recall', 0):.3f}, "
                       f"F1={metrics.get('test_f1', 0):.3f}")
        else:
            logger.info(f"{config_name}: Failed - {metrics['error'][:50]}...")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ All combinations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    return all_metrics


if __name__ == "__main__":
    main()