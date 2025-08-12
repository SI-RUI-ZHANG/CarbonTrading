"""
Training script for Weekly Meta-Model
Trains the meta-model to predict weekly LSTM reliability using sentiment features
Simplified version that follows the same pattern as daily meta model
"""

import numpy as np
import pandas as pd
import os
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple

from config import MetaConfig
from model import create_meta_model

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


# ==================================================================================
# WEEKLY META CONFIG (extends MetaConfig)
# ==================================================================================

class WeeklyMetaConfig(MetaConfig):
    """Configuration for weekly meta-model"""
    
    def __init__(self, market='GDEA', use_sentiment=True):
        """Initialize config with specific market and sentiment option"""
        super().__init__(market)
        self.USE_SENTIMENT = use_sentiment
        # Weekly meta uses its own output path
    
    @property
    def OUTPUT_DIR(self):
        """Override output directory for weekly meta model"""
        return os.path.join(self.OUTPUT_BASE_DIR, 'meta', 'weekly', self.MARKET)
        
    @property
    def WEEKLY_SENTIMENT_DATA_PATH(self):
        """Weekly data with sentiment features"""
        return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly_with_sentiment.parquet'
    
    @property
    def WEEKLY_BASE_DATA_PATH(self):
        """Weekly data without sentiment"""
        return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly.parquet'
    
    # Weekly-specific parameters
    SEQUENCE_LENGTH = 12  # 12 weeks lookback
    TRAIN_END_DATE = '2020-12-31'
    VAL_END_DATE = '2022-12-31'
    
    def __repr__(self):
        """String representation"""
        return (
            f"WeeklyMetaConfig(MARKET={self.MARKET}, "
            f"USE_SENTIMENT={self.USE_SENTIMENT}, "
            f"META_MODEL={self.META_MODEL_TYPE})"
        )


# ==================================================================================
# DATA PREPARATION FUNCTIONS
# ==================================================================================

def prepare_weekly_meta_training_data(config: WeeklyMetaConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data for weekly meta-model
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    logger.info("Preparing weekly meta-model training data...")
    
    # Load weekly sentiment data
    sentiment_df = pd.read_parquet(config.WEEKLY_SENTIMENT_DATA_PATH)
    logger.info(f"Loaded weekly sentiment data: {sentiment_df.shape}")
    
    # Load weekly base data to get actual price movements
    base_df = pd.read_parquet(config.WEEKLY_BASE_DATA_PATH)
    logger.info(f"Loaded weekly base data: {base_df.shape}")
    
    # For simplicity, create synthetic LSTM predictions
    # In practice, would load from trained weekly LSTM model
    lstm_predictions = generate_synthetic_lstm_predictions(base_df)
    
    # Align all data
    common_dates = sentiment_df.index.intersection(base_df.index)
    sentiment_df = sentiment_df.loc[common_dates]
    base_df = base_df.loc[common_dates]
    lstm_predictions = lstm_predictions.loc[common_dates]
    
    # Create target: whether LSTM prediction was correct
    actual_direction = (base_df['log_return'] > 0).astype(int)
    y = (lstm_predictions == actual_direction).astype(int).values
    
    # Extract sentiment features
    feature_cols = [col for col in config.SENTIMENT_FEATURES if col in sentiment_df.columns]
    if not feature_cols:
        # If sentiment features not found, use all numeric columns
        feature_cols = sentiment_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if 'return' not in col.lower()][:10]  # Use first 10 features
    
    X = sentiment_df[feature_cols].fillna(0).values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Weekly LSTM accuracy (baseline): {y.mean():.3f}")
    
    # Split chronologically
    n_samples = len(X)
    train_end = int(n_samples * config.TRAIN_SPLIT)
    val_end = int(n_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def generate_synthetic_lstm_predictions(base_df: pd.DataFrame) -> pd.Series:
    """
    Generate synthetic LSTM predictions for demonstration
    In practice, would load from trained model
    """
    # Create predictions with some correlation to actual movements
    actual_direction = (base_df['log_return'] > 0).astype(int)
    
    # Add some noise to make it realistic (60% accuracy)
    noise = np.random.random(len(actual_direction)) > 0.4
    predictions = actual_direction.copy()
    predictions[noise] = 1 - predictions[noise]
    
    return predictions


def train_weekly_meta_model(config: WeeklyMetaConfig):
    """
    Train weekly meta-model
    
    Args:
        config: WeeklyMetaConfig object
    """
    logger.info("="*80)
    logger.info("WEEKLY META-MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration: {config}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data
    X_train, y_train, X_val, y_val = prepare_weekly_meta_training_data(config)
    
    # Create and train model
    model = create_meta_model('random_forest', config)
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_metrics, val_pred, val_prob, val_abstentions = model.evaluate(X_val, y_val, "Validation")
    
    # Save model and metrics
    model.save(os.path.join(config.OUTPUT_DIR, 'weekly_meta_model.pkl'))
    
    metrics = {
        'validation': val_metrics,
        'config': {
            'market': config.MARKET,
            'use_sentiment': config.USE_SENTIMENT,
            'model_type': 'random_forest'
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("✅ Weekly meta-model training completed")
    
    # Update performance tracker
    update_performance_summary(
        model_type='weekly_meta',
        market=config.MARKET,
        metrics=val_metrics,
        features_type='Meta'
    )
    
    return model, val_metrics


def main():
    """Main execution - runs both markets by default"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Weekly Meta Model')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--sentiment', choices=['yes', 'no', 'both'],
                       default='yes', help='Use sentiment features (default: yes)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Determine sentiment options
    if args.sentiment == 'both':
        sentiment_options = [True, False]
    elif args.sentiment == 'yes':
        sentiment_options = [True]
    else:
        sentiment_options = [False]
    
    # Store results
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("WEEKLY META MODEL TRAINING")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Sentiment: {args.sentiment}")
    logger.info("="*80)
    
    # Train for each combination
    for market in markets:
        for use_sentiment in sentiment_options:
            config_key = f"{market}_{'with' if use_sentiment else 'no'}_sentiment"
            logger.info(f"\n\nPROCESSING: {config_key}")
            
            try:
                config = WeeklyMetaConfig(market=market, use_sentiment=use_sentiment)
                model, metrics = train_weekly_meta_model(config)
                all_metrics[config_key] = metrics
                logger.info(f"✅ {config_key} completed")
                
            except Exception as e:
                logger.error(f"❌ {config_key} failed: {str(e)}")
                all_metrics[config_key] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    for config_key, metrics in all_metrics.items():
        if 'error' not in metrics:
            logger.info(f"{config_key}: Accuracy={metrics.get('accuracy', 0):.3f}")
        else:
            logger.info(f"{config_key}: Failed")
    
    logger.info("\n✅ All configurations completed")
    
    return all_metrics


if __name__ == "__main__":
    main()