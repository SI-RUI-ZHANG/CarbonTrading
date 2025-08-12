"""
Training script for Meta Model
Identifies and reverses LSTM errors to improve accuracy.
No abstention - always makes predictions (100% coverage).
"""

import numpy as np
import pandas as pd
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple

from config import ReversalConfig
from error_reversal_model import ErrorReversalMetaModel

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


def train_error_reversal_model(config=None):
    """
    Main training function for meta model
    
    Args:
        config: ReversalConfig object (uses default if None)
    """
    if config is None:
        config = ReversalConfig()
    
    logger.info("="*80)
    logger.info("META MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration: {config}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save configuration
    config.save(os.path.join(config.output_dir, 'config.json'))
    
    # ==================================================================================
    # STEP 1: PREPARE DATA
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("-"*60)
    
    # Check if data already exists
    data_files = ['X_train.npy', 'lstm_preds_train.npy', 'y_train.npy',
                  'X_val.npy', 'lstm_preds_val.npy', 'y_val.npy']
    data_exists = all(os.path.exists(os.path.join(config.output_dir, f)) for f in data_files)
    
    if data_exists:
        logger.info("Loading existing prepared data...")
        X_train = np.load(os.path.join(config.output_dir, 'X_train.npy'))
        lstm_preds_train = np.load(os.path.join(config.output_dir, 'lstm_preds_train.npy'))
        y_train = np.load(os.path.join(config.output_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(config.output_dir, 'X_val.npy'))
        lstm_preds_val = np.load(os.path.join(config.output_dir, 'lstm_preds_val.npy'))
        y_val = np.load(os.path.join(config.output_dir, 'y_val.npy'))
    else:
        logger.info("Preparing new data...")
        X_train, lstm_preds_train, y_train, X_val, lstm_preds_val, y_val = prepare_error_reversal_data(config)
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    
    # Calculate baseline LSTM accuracy
    lstm_train_acc = (lstm_preds_train == y_train).mean()
    lstm_val_acc = (lstm_preds_val == y_val).mean()
    logger.info(f"LSTM baseline - Train: {lstm_train_acc:.1%}, Val: {lstm_val_acc:.1%}")
    
    # ==================================================================================
    # STEP 2: TRAIN ERROR REVERSAL MODEL
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("-"*60)
    
    # Create error reversal model
    model = ErrorReversalMetaModel(config)
    
    # Train model
    model.train(X_train, lstm_preds_train, y_train, 
                X_val, lstm_preds_val, y_val)
    
    # ==================================================================================
    # STEP 3: EVALUATE PERFORMANCE
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 3: EVALUATION")
    logger.info("-"*60)
    
    # Evaluate on training set
    train_metrics, train_final, train_error_probs = model.evaluate(
        X_train, lstm_preds_train, y_train, "Training"
    )
    
    # Evaluate on validation set
    val_metrics, val_final, val_error_probs = model.evaluate(
        X_val, lstm_preds_val, y_val, "Validation"
    )
    
    # ==================================================================================
    # STEP 4: ANALYZE REVERSAL IMPACT
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 4: REVERSAL ANALYSIS")
    logger.info("-"*60)
    
    analyze_reversal_impact(val_error_probs, lstm_preds_val, val_final, y_val, config)
    
    # ==================================================================================
    # STEP 5: SAVE RESULTS
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 5: SAVING RESULTS")
    logger.info("-"*60)
    
    # Save model
    model.save(os.path.join(config.output_dir, 'error_reversal_model.pkl'))
    
    # Save metrics (convert numpy types to Python types for JSON)
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metrics = {
        'training': convert_to_json_serializable(train_metrics),
        'validation': convert_to_json_serializable(val_metrics),
        'reversal_stats': convert_to_json_serializable(model.reversal_stats),
        'config': {
            'market': config.MARKET,
            'frequency': config.FREQUENCY,
            'reversal_threshold': float(model.reversal_threshold),
            'coverage': 1.0
        }
    }
    
    with open(os.path.join(config.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    np.save(os.path.join(config.output_dir, 'val_predictions.npy'), val_final)
    np.save(os.path.join(config.output_dir, 'val_error_probs.npy'), val_error_probs)
    
    # ==================================================================================
    # STEP 6: CREATE VISUALIZATIONS
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 6: CREATING VISUALIZATIONS")
    logger.info("-"*60)
    
    create_reversal_visualizations(model, val_error_probs, lstm_preds_val, 
                                  val_final, y_val, val_metrics, config)
    
    logger.info("Visualizations saved")
    
    # ==================================================================================
    # FINAL SUMMARY
    # ==================================================================================
    
    logger.info("\n" + "="*80)
    logger.info("✅ META MODEL TRAINING COMPLETED")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("="*80)
    
    # Update performance tracker
    model_type = f'{config.FREQUENCY}_meta'
    update_performance_summary(
        model_type=model_type,
        market=config.MARKET,
        metrics={
            'accuracy': val_metrics['final_accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1_score': val_metrics['f1_score'],
            'test_samples': val_metrics['total_samples'],
            # Meta-specific metrics
            'reversal_rate': val_metrics['reversal_rate'],
            'improvement': val_metrics['improvement'],
            'coverage': 1.0,  # Always 100%
            'abstention_rate': 0.0  # Always 0% - no abstention
        },
        features_type='Meta Model'
    )
    
    return model, val_metrics


def prepare_error_reversal_data(config) -> Tuple:
    """
    Prepare training data for error reversal meta-model
    
    Returns:
        Tuple of (X_train, lstm_preds_train, y_train, X_val, lstm_preds_val, y_val)
    """
    logger.info("Preparing error reversal training data...")
    
    # Load sentiment data
    sentiment_df = pd.read_parquet(config.SENTIMENT_DATA_PATH)
    logger.info(f"Loaded sentiment data: {sentiment_df.shape}")
    
    # Load base data (without sentiment) to get actual price movements
    base_df = pd.read_parquet(config.BASE_DATA_PATH)
    logger.info(f"Loaded base data: {base_df.shape}")
    
    # Load primary LSTM model predictions and probabilities
    lstm_predictions, lstm_probabilities = load_lstm_predictions(config, base_df)
    
    # Align all data
    common_dates = sentiment_df.index.intersection(base_df.index).intersection(lstm_predictions.index)
    sentiment_df = sentiment_df.loc[common_dates]
    base_df = base_df.loc[common_dates]
    lstm_predictions = lstm_predictions.loc[common_dates]
    lstm_probabilities = lstm_probabilities.loc[common_dates]
    logger.info(f"Common dates after alignment: {len(common_dates)}")
    
    # Get actual direction (target)
    actual_direction = (base_df['log_return'] > 0).astype(int)
    
    # Extract sentiment features
    feature_cols = [col for col in config.SENTIMENT_FEATURES if col in sentiment_df.columns]
    sentiment_features = sentiment_df[feature_cols]
    
    # Create feature matrix with LSTM and error detection features
    feature_df = pd.DataFrame(index=common_dates)
    
    # Add LSTM features
    feature_df['lstm_prediction'] = lstm_predictions.values
    feature_df['lstm_probability'] = lstm_probabilities.values
    feature_df['lstm_confidence'] = np.abs(lstm_probabilities.values - 0.5) * 2
    feature_df['lstm_entropy'] = -(
        lstm_probabilities.values * np.log(lstm_probabilities.values + 1e-8) + 
        (1 - lstm_probabilities.values) * np.log(1 - lstm_probabilities.values + 1e-8)
    )
    
    # Add sentiment features
    for col in feature_cols:
        feature_df[col] = sentiment_features[col].values
    
    # Add error detection features
    feature_df['lstm_uncertainty'] = (np.abs(lstm_probabilities.values - 0.5) < 0.1).astype(int)
    feature_df['lstm_extreme_conf'] = ((lstm_probabilities.values > 0.8) | (lstm_probabilities.values < 0.2)).astype(int)
    
    if 'sentiment_mean' in sentiment_features.columns:
        sentiment_signal = (sentiment_features['sentiment_mean'] > 0).astype(int)
        feature_df['sentiment_agrees_with_lstm'] = (sentiment_signal == lstm_predictions.values).astype(int)
        feature_df['sentiment_strong_disagree'] = (
            (np.abs(sentiment_features['sentiment_mean'].values) > 20) & 
            (sentiment_signal != lstm_predictions.values)
        ).astype(int)
        feature_df['confidence_sentiment_interaction'] = lstm_probabilities.values * sentiment_features['sentiment_mean'].values
    
    # Add market volatility if available
    if 'volatility' in base_df.columns:
        feature_df['market_volatility'] = base_df['volatility'].values
    
    X = feature_df.values
    lstm_preds = lstm_predictions.values
    y = actual_direction.values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Feature names: {list(feature_df.columns)}")
    
    # Split chronologically
    n_samples = len(X)
    train_end = int(n_samples * config.TRAIN_SPLIT)
    val_end = int(n_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    X_train = X[:train_end]
    lstm_preds_train = lstm_preds[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    lstm_preds_val = lstm_preds[train_end:val_end]
    y_val = y[train_end:val_end]
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    
    # Save prepared data
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.OUTPUT_DIR, 'lstm_preds_train.npy'), lstm_preds_train)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.OUTPUT_DIR, 'lstm_preds_val.npy'), lstm_preds_val)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_val.npy'), y_val)
    
    return X_train, lstm_preds_train, y_train, X_val, lstm_preds_val, y_val


def load_lstm_predictions(config, base_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Load LSTM model predictions and probabilities from the primary model
    
    Args:
        config: ReversalConfig object
        base_df: Base data DataFrame
        
    Returns:
        Tuple of (predictions Series, probabilities Series)
    """
    model_dir = config.PRIMARY_MODEL_DIR
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Primary LSTM model not found at: {model_dir}. Please train the daily LSTM model first.")
    
    logger.info(f"Using primary LSTM model from: {model_dir}")
    
    # Load predictions, actuals, and probabilities
    pred_file = os.path.join(model_dir, 'test_predictions.npy')
    prob_file = os.path.join(model_dir, 'test_probabilities.npy')
    
    if not os.path.exists(pred_file):
        raise ValueError(f"Predictions file not found: {pred_file}. Please train the daily LSTM model first.")
    
    # Load predictions
    predictions = np.load(pred_file)
    
    # Load probabilities (if available, otherwise use predictions as proxy)
    if os.path.exists(prob_file):
        probabilities = np.load(prob_file)
        logger.info("Loaded LSTM probabilities")
    else:
        # Fallback: use predictions as confidence (0 or 1)
        logger.warning("Probabilities file not found. Using binary predictions as proxy.")
        logger.warning("Re-train LSTM model to generate probability outputs for better meta-model performance.")
        probabilities = predictions.astype(float)
    
    # Create series aligned with the last part of base_df (test period)
    pred_series = pd.Series(predictions, index=base_df.index[-len(predictions):])
    prob_series = pd.Series(probabilities, index=base_df.index[-len(probabilities):])
    
    logger.info(f"Loaded {len(predictions)} LSTM predictions and probabilities from test set")
    
    return pred_series, prob_series


def analyze_reversal_impact(error_probs, lstm_preds, final_preds, y_true, config):
    """
    Analyze the impact of reversals on performance
    
    Args:
        error_probs: Probability that LSTM is wrong
        lstm_preds: Original LSTM predictions
        final_preds: Final predictions after reversals
        y_true: True labels
        config: ReversalConfig object
    """
    reversals = (lstm_preds != final_preds)
    reversal_rate = reversals.mean()
    
    logger.info(f"\nReversal Impact Analysis:")
    logger.info(f"  Total reversals: {reversals.sum()}/{len(reversals)} ({reversal_rate:.1%})")
    
    if reversals.any():
        # Analyze reversed predictions
        reversed_correct = (final_preds[reversals] == y_true[reversals]).mean()
        logger.info(f"  Reversal success rate: {reversed_correct:.1%}")
        
        # Analyze by confidence buckets
        logger.info("\nReversals by Error Probability:")
        bins = [0.5, 0.6, 0.7, 0.8, 1.0]
        for i in range(len(bins)-1):
            mask = (error_probs >= bins[i]) & (error_probs < bins[i+1]) & reversals
            if mask.sum() > 0:
                success = (final_preds[mask] == y_true[mask]).mean()
                logger.info(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): "
                           f"{mask.sum()} reversals, {success:.1%} correct")


def create_reversal_visualizations(model, error_probs, lstm_preds, final_preds, 
                                  y_true, metrics, config):
    """
    Create visualization plots for error reversal analysis
    
    Args:
        model: Trained ErrorReversalMetaModel
        error_probs: Error probabilities
        lstm_preds: Original LSTM predictions
        final_preds: Final predictions after reversals
        y_true: True labels
        metrics: Validation metrics
        config: ReversalConfig object
    """
    reversals = (lstm_preds != final_preds)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Error probability distribution
    axes[0, 0].hist(error_probs, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(model.reversal_threshold, color='red', 
                       linestyle='--', label=f'Threshold ({model.reversal_threshold:.2f})')
    axes[0, 0].set_xlabel('LSTM Error Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Error Probabilities')
    axes[0, 0].legend()
    
    # 2. Accuracy comparison
    accuracies = pd.DataFrame({
        'LSTM Baseline': [metrics['lstm_baseline_accuracy']],
        'After Reversals': [metrics['final_accuracy']]
    })
    accuracies.plot(kind='bar', ax=axes[0, 1], color=['blue', 'green'])
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'Accuracy Improvement: {metrics["improvement"]:+.1%}')
    axes[0, 1].set_xticklabels([''], rotation=0)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reversal analysis
    reversal_counts = [reversals.sum(), (~reversals).sum()]
    reversal_labels = ['Reversed', 'Not Reversed']
    axes[1, 0].pie(reversal_counts, labels=reversal_labels, 
                   autopct='%1.1f%%', colors=['orange', 'lightblue'])
    axes[1, 0].set_title(f'Reversal Rate: {metrics["reversal_rate"]:.1%}')
    axes[1, 0].set_ylabel('')
    
    # 4. Feature importance (if available)
    if hasattr(model, 'feature_importance') and model.feature_importance is not None:
        feature_names = config.LSTM_FEATURES[:4] + ['sent_supply', 'sent_demand', 
                                                    'sent_momentum', 'lstm_uncert', 
                                                    'agrees', 'disagrees']
        importances = model.feature_importance[:10]
        y_pos = np.arange(len(importances))
        axes[1, 1].barh(y_pos, importances, color='steelblue')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(feature_names[:len(importances)])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top Features for Error Detection')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center')
        axes[1, 1].set_title('Feature Importance')
    
    plt.suptitle(f'Error Reversal Analysis - {config.MARKET}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(config.output_dir, 'error_reversal_analysis.png'), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved")


def main():
    """Main execution - runs combinations of markets and frequencies"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Meta Model')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--frequency', choices=['daily', 'weekly', 'both'],
                       default='both', help='Frequency(ies) to run (default: both)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Determine which frequencies to run
    if args.frequency == 'both':
        frequencies = ['daily', 'weekly']
    else:
        frequencies = [args.frequency]
    
    # Store results for all combinations
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("META MODEL TRAINING")
    logger.info(f"Markets to process: {', '.join(markets)}")
    logger.info(f"Frequencies to process: {', '.join(frequencies)}")
    logger.info("Coverage: 100% (No abstention!)")
    logger.info("="*80)
    
    # Train model for each combination
    for frequency in frequencies:
        for market in markets:
            config_name = f"{frequency}_{market}"
            logger.info(f"\n\nPROCESSING: {config_name}")
            
            try:
                config = ReversalConfig(market=market, frequency=frequency)
                model, metrics = train_error_reversal_model(config)
                all_metrics[config_name] = metrics
                logger.info(f"✅ {config_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {config_name} failed: {e}")
                all_metrics[config_name] = None
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    for config_name, metrics in all_metrics.items():
        if metrics:
            logger.info(f"{config_name}: Improvement={metrics['improvement']:+.1%}, "
                       f"Reversal Rate={metrics['reversal_rate']:.1%}, "
                       f"Final Accuracy={metrics['final_accuracy']:.1%}")
        else:
            logger.info(f"{config_name}: Failed")
    
    logger.info("\n" + "="*80)
    logger.info("✅ All combinations completed")
    logger.info("="*80)
    
    return all_metrics


if __name__ == "__main__":
    main()