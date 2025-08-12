#!/usr/bin/env python3
"""
Unified Meta Model Runner
Automatically runs both daily and weekly meta models for all markets
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Import the individual runners
from run import train_meta_model, MetaConfig
from run_weekly import train_weekly_meta_model, WeeklyMetaConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_all_meta_models(markets=None, model_types=None):
    """
    Run all meta model configurations
    
    Args:
        markets: List of markets to run (default: ['GDEA', 'HBEA'])
        model_types: List of model types to run (default: ['daily', 'weekly'])
    """
    if markets is None:
        markets = ['GDEA', 'HBEA']
    if model_types is None:
        model_types = ['daily', 'weekly']
    
    # Store all results
    all_results = {}
    start_time = datetime.now()
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE META MODEL TRAINING")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Model Types: {', '.join(model_types)}")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Counter for progress
    total_configs = len(markets) * len(model_types)
    current_config = 0
    
    # Run each configuration
    for model_type in model_types:
        for market in markets:
            current_config += 1
            config_name = f"{model_type}_{market}"
            
            logger.info(f"\n[{current_config}/{total_configs}] Processing: {config_name}")
            logger.info("-"*60)
            
            try:
                if model_type == 'daily':
                    # Run daily meta model
                    config = MetaConfig(market=market)
                    model, metrics = train_meta_model(config)
                    all_results[config_name] = {
                        'status': 'success',
                        'metrics': metrics,
                        'model_type': 'daily',
                        'market': market
                    }
                    logger.info(f"✅ Daily meta model for {market} completed")
                    
                elif model_type == 'weekly':
                    # Run weekly meta model (with sentiment by default)
                    config = WeeklyMetaConfig(market=market, use_sentiment=True)
                    model, metrics = train_weekly_meta_model(config)
                    all_results[config_name] = {
                        'status': 'success',
                        'metrics': metrics,
                        'model_type': 'weekly',
                        'market': market
                    }
                    logger.info(f"✅ Weekly meta model for {market} completed")
                    
            except Exception as e:
                logger.error(f"❌ {config_name} failed: {str(e)}")
                all_results[config_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'model_type': model_type,
                    'market': market
                }
    
    # Calculate runtime
    runtime = datetime.now() - start_time
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    
    successful = sum(1 for r in all_results.values() if r['status'] == 'success')
    failed = sum(1 for r in all_results.values() if r['status'] == 'failed')
    
    logger.info(f"Total Configurations: {total_configs}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful/total_configs*100:.1f}%")
    logger.info(f"Total Runtime: {runtime}")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    logger.info("-"*40)
    
    for config_name, result in all_results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            logger.info(f"\n{config_name}:")
            logger.info(f"  Status: ✅ Success")
            if 'accuracy' in metrics:
                logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            if 'coverage' in metrics:
                logger.info(f"  Coverage: {metrics['coverage']:.1%}")
            if 'abstention_rate' in metrics:
                logger.info(f"  Abstention Rate: {metrics['abstention_rate']:.1%}")
        else:
            logger.info(f"\n{config_name}:")
            logger.info(f"  Status: ❌ Failed")
            logger.info(f"  Error: {result['error'][:100]}...")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL META MODEL TRAINING COMPLETED")
    logger.info("="*80)
    
    return all_results


def main():
    """Main execution function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run all meta models for daily and weekly LSTM predictions'
    )
    parser.add_argument(
        '--markets',
        nargs='+',
        choices=['GDEA', 'HBEA'],
        default=['GDEA', 'HBEA'],
        help='Markets to process (default: both)'
    )
    parser.add_argument(
        '--types',
        nargs='+',
        choices=['daily', 'weekly'],
        default=['daily', 'weekly'],
        help='Model types to run (default: both)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode - runs GDEA daily only'
    )
    
    args = parser.parse_args()
    
    # Quick mode for testing
    if args.quick:
        logger.info("Running in QUICK TEST mode (GDEA daily only)")
        markets = ['GDEA']
        model_types = ['daily']
    else:
        markets = args.markets
        model_types = args.types
    
    # Run all configurations
    results = run_all_meta_models(markets=markets, model_types=model_types)
    
    # Return exit code based on results
    failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())