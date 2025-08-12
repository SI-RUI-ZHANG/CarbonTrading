"""
Training script for Meta-Model
Trains the meta-model to predict LSTM reliability using sentiment features
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


def train_meta_model(config=None):
    """
    Main training function for meta-model
    
    Args:
        config: MetaConfig object (uses default if None)
    """
    if config is None:
        config = MetaConfig()
    
    logger.info("="*80)
    logger.info("META-MODEL TRAINING")
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
    data_files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy']
    data_exists = all(os.path.exists(os.path.join(config.output_dir, f)) for f in data_files)
    
    if data_exists:
        logger.info("Loading existing prepared data...")
        X_train = np.load(os.path.join(config.output_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(config.output_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(config.output_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(config.output_dir, 'y_val.npy'))
    else:
        logger.info("Preparing new data...")
        X_train, y_train, X_val, y_val = prepare_meta_training_data(config)
    
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Class balance - Train: {y_train.mean():.1%} correct")
    logger.info(f"Class balance - Val: {y_val.mean():.1%} correct")
    
    # ==================================================================================
    # STEP 2: TRAIN META-MODEL
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("-"*60)
    
    # Create Random Forest model with abstention
    # Now always uses Random Forest regardless of config
    model = create_meta_model('random_forest', config)
    
    # Train model with validation data for monitoring
    model.train(X_train, y_train, X_val, y_val)
    
    # ==================================================================================
    # STEP 3: EVALUATE PERFORMANCE
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 3: EVALUATION")
    logger.info("-"*60)
    
    # Evaluate on training set with abstention
    train_metrics, train_pred, train_prob, train_abstentions = model.evaluate(X_train, y_train, "Training")
    
    # Evaluate on validation set with abstention
    val_metrics, val_pred, val_prob, val_abstentions = model.evaluate(X_val, y_val, "Validation")
    
    # ==================================================================================
    # STEP 4: ANALYZE ENSEMBLE PERFORMANCE
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 4: ENSEMBLE ANALYSIS")
    logger.info("-"*60)
    
    # Simulate ensemble trading strategy
    analyze_ensemble_performance(val_prob, y_val, config)
    
    # ==================================================================================
    # STEP 5: SAVE RESULTS
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 5: SAVING RESULTS")
    logger.info("-"*60)
    
    # Save Random Forest model
    model.save(os.path.join(config.output_dir, 'meta_model_rf.pkl'))
    
    # Save metrics including abstention statistics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics,
        'abstention_stats': {
            'train_abstention_rate': train_abstentions.mean() if train_abstentions is not None else 0,
            'val_abstention_rate': val_abstentions.mean() if val_abstentions is not None else 0,
            'train_coverage': 1 - (train_abstentions.mean() if train_abstentions is not None else 0),
            'val_coverage': 1 - (val_abstentions.mean() if val_abstentions is not None else 0)
        },
        'oob_score': model.oob_score,
        'config': {
            'model_type': 'random_forest',
            'market': config.MARKET,
            'strategy': config.ENSEMBLE_STRATEGY,
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'abstention_threshold': config.ABSTENTION_THRESHOLD,
            'min_sentiment_coverage': config.MIN_SENTIMENT_COVERAGE
        }
    }
    
    with open(os.path.join(config.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and abstentions
    np.save(os.path.join(config.output_dir, 'val_predictions.npy'), val_pred)
    np.save(os.path.join(config.output_dir, 'val_probabilities.npy'), val_prob)
    np.save(os.path.join(config.output_dir, 'val_abstentions.npy'), val_abstentions)
    
    # ==================================================================================
    # STEP 6: VISUALIZATIONS
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 6: CREATING VISUALIZATIONS")
    logger.info("-"*60)
    
    # Create visualizations
    create_visualizations(model, val_prob, y_val, config)
    
    logger.info("\n" + "="*80)
    logger.info("✅ META-MODEL TRAINING COMPLETED")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("="*80)
    
    # Update performance tracker
    update_performance_summary(
        model_type='daily_meta',
        market=config.MARKET,
        metrics=val_metrics,
        features_type='Meta'
    )
    logger.info("Performance summary updated")
    
    return model, val_metrics


def analyze_ensemble_performance(probabilities, true_correctness, config):
    """
    Analyze how the ensemble strategy would perform
    
    Args:
        probabilities: Meta-model confidence scores
        true_correctness: Whether LSTM predictions were actually correct
        config: MetaConfig object
    """
    # Baseline: Trust all LSTM predictions
    baseline_accuracy = true_correctness.mean()
    logger.info(f"\nBaseline (trust all): {baseline_accuracy:.1%} accuracy")
    
    # Filtered strategy: Only trust high-confidence predictions
    high_conf_mask = probabilities > config.CONFIDENCE_THRESHOLD
    
    if high_conf_mask.sum() > 0:
        filtered_accuracy = true_correctness[high_conf_mask].mean()
        coverage = high_conf_mask.mean()
        
        logger.info(f"\nFiltered Strategy (threshold={config.CONFIDENCE_THRESHOLD}):")
        logger.info(f"  Coverage: {coverage:.1%} of predictions")
        logger.info(f"  Accuracy when trading: {filtered_accuracy:.1%}")
        logger.info(f"  Improvement: {(filtered_accuracy - baseline_accuracy):.1%}")
    else:
        logger.warning("No predictions above confidence threshold!")
    
    # Analyze by confidence buckets
    logger.info("\nAccuracy by Confidence Level:")
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    for i in range(len(bins)-1):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i+1])
        if mask.sum() > 0:
            bucket_acc = true_correctness[mask].mean()
            logger.info(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): "
                       f"{bucket_acc:.1%} ({mask.sum()} samples)")


def create_visualizations(model, probabilities, true_correctness, config):
    """
    Create visualization plots including abstention analysis
    
    Args:
        model: Trained Random Forest meta-model
        probabilities: Confidence scores
        true_correctness: Actual correctness
        config: MetaConfig object
    """
    # Create abstention analysis plot if using Random Forest
    if hasattr(model, 'plot_abstention_analysis'):
        # Need X and y for abstention analysis
        # For now, we'll skip this plot unless we have the data
        pass
    
    # Plot feature importance for Random Forest
    if hasattr(model, 'plot_feature_importance'):
        model.plot_feature_importance(
            save_path=os.path.join(config.output_dir, 'feature_importance.png')
        )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Confidence distribution
    axes[0, 0].hist(probabilities, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(config.CONFIDENCE_THRESHOLD, color='red', 
                       linestyle='--', label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')
    axes[0, 0].set_xlabel('Meta-Model Confidence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Confidence Scores')
    axes[0, 0].legend()
    
    # 2. Calibration plot
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (probabilities >= bins[i]) & (probabilities < bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(true_correctness[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    axes[0, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, edgecolor='black')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Actual Accuracy')
    axes[0, 1].set_title('Calibration Plot')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Coverage vs Accuracy trade-off
    thresholds = np.linspace(0, 1, 50)
    coverages = []
    accuracies = []
    
    for thresh in thresholds:
        mask = probabilities >= thresh
        if mask.sum() > 0:
            coverages.append(mask.mean())
            accuracies.append(true_correctness[mask].mean())
        else:
            coverages.append(0)
            accuracies.append(0)
    
    axes[1, 0].plot(coverages, accuracies, 'b-', linewidth=2)
    axes[1, 0].axhline(true_correctness.mean(), color='gray', 
                       linestyle='--', label='Baseline')
    axes[1, 0].set_xlabel('Coverage (% of predictions used)')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Coverage vs Accuracy Trade-off')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature importance (if XGBoost)
    if config.META_MODEL_TYPE == 'xgboost' and hasattr(model, 'feature_importance'):
        feature_names = config.SENTIMENT_FEATURES + ['lstm_pred', 'day_of_week']
        
        # Get importance scores
        importance_dict = model.feature_importance
        features = []
        importances = []
        
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:8]:
            feat_idx = int(feat[1:]) if feat.startswith('f') else 0
            if feat_idx < len(feature_names):
                features.append(feature_names[feat_idx])
            else:
                features.append(feat)
            importances.append(imp)
        
        y_pos = np.arange(len(features))
        axes[1, 1].barh(y_pos, importances, color='steelblue')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(features)
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Top Feature Importances')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                       ha='center', va='center')
        axes[1, 1].set_title('Feature Importance')
    
    plt.suptitle(f'Meta-Model Analysis - {config.MARKET}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(config.output_dir, 'meta_model_analysis.png'), 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved")


# ==================================================================================
# DATA PREPARATION FUNCTIONS (integrated from data_preparation.py)
# ==================================================================================

def prepare_meta_training_data(config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data for meta-model
    
    The meta-model learns to predict when the primary LSTM model will be correct,
    using sentiment features as input.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
        Where X contains sentiment features and y contains correctness labels
    """
    logger.info("Preparing meta-model training data...")
    
    # Load sentiment data
    sentiment_df = pd.read_parquet(config.SENTIMENT_DATA_PATH)
    logger.info(f"Loaded sentiment data: {sentiment_df.shape}")
    
    # Load base data (without sentiment) to get actual price movements
    base_df = pd.read_parquet(config.BASE_DATA_PATH)
    logger.info(f"Loaded base data: {base_df.shape}")
    
    # Find or load primary LSTM model predictions and probabilities
    lstm_predictions, lstm_probabilities = load_lstm_predictions(config, base_df)
    
    # Align all data
    common_dates = sentiment_df.index.intersection(base_df.index).intersection(lstm_predictions.index)
    sentiment_df = sentiment_df.loc[common_dates]
    base_df = base_df.loc[common_dates]
    lstm_predictions = lstm_predictions.loc[common_dates]
    lstm_probabilities = lstm_probabilities.loc[common_dates]
    logger.info(f"Common dates after alignment: {len(common_dates)}")
    
    # Create target: whether LSTM prediction was correct
    actual_direction = (base_df['log_return'] > 0).astype(int)
    y = (lstm_predictions == actual_direction).astype(int).values
    
    # Extract sentiment features
    feature_cols = [col for col in config.SENTIMENT_FEATURES if col in sentiment_df.columns]
    sentiment_features = sentiment_df[feature_cols]
    
    # Create feature matrix with LSTM predictions and probabilities
    feature_df = pd.DataFrame(index=common_dates)
    
    # Add LSTM features (CRITICAL FIX)
    feature_df['lstm_prediction'] = lstm_predictions.values
    feature_df['lstm_probability'] = lstm_probabilities.values
    feature_df['lstm_confidence'] = np.abs(lstm_probabilities.values - 0.5) * 2  # Distance from 0.5
    feature_df['lstm_entropy'] = -(
        lstm_probabilities.values * np.log(lstm_probabilities.values + 1e-8) + 
        (1 - lstm_probabilities.values) * np.log(1 - lstm_probabilities.values + 1e-8)
    )
    
    # Add sentiment features
    for col in feature_cols:
        feature_df[col] = sentiment_features[col].values
    
    # Add interaction features
    if 'sentiment_mean' in sentiment_features.columns:
        # Calculate sentiment signal (1 if positive, 0 if negative)
        sentiment_signal = (sentiment_features['sentiment_mean'] > 0).astype(int)
        feature_df['sentiment_agrees_with_lstm'] = (sentiment_signal == lstm_predictions.values).astype(int)
        feature_df['confidence_sentiment_interaction'] = lstm_probabilities.values * sentiment_features['sentiment_mean'].values
    
    # Add market volatility (if available)
    if 'volatility' in base_df.columns:
        feature_df['market_volatility'] = base_df['volatility'].values
    
    X = feature_df.values
    
    logger.info(f"Enhanced features with LSTM information")
    logger.info(f"Feature names: {list(feature_df.columns)}")
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"LSTM accuracy (baseline): {y.mean():.3f}")
    
    # Split chronologically (using indices to match original LSTM splits)
    n_samples = len(X)
    train_end = int(n_samples * config.TRAIN_SPLIT)
    val_end = int(n_samples * (config.TRAIN_SPLIT + config.VAL_SPLIT))
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    
    # Save prepared data
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_val.npy'), y_val)
    
    return X_train, y_train, X_val, y_val


def load_lstm_predictions(config, base_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Load LSTM model predictions and probabilities from the primary model
    
    Args:
        config: MetaConfig object
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
    actual_file = os.path.join(model_dir, 'test_actuals.npy')
    
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
    # The test predictions correspond to the last len(predictions) dates
    pred_series = pd.Series(predictions, index=base_df.index[-len(predictions):])
    prob_series = pd.Series(probabilities, index=base_df.index[-len(probabilities):])
    
    logger.info(f"Loaded {len(predictions)} LSTM predictions and probabilities from test set")
    
    return pred_series, prob_series


def main():
    """Main execution - runs both markets by default"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Daily Meta Model')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--all', action='store_true',
                       help='Run complete meta model suite (daily + weekly for all markets)')
    args = parser.parse_args()
    
    # If --all flag is set, run the comprehensive suite
    if args.all:
        logger.info("Running complete meta model suite...")
        from run_all import run_all_meta_models
        results = run_all_meta_models()
        return results
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Store results for all markets
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("META MODEL TRAINING FOR DAILY LSTM")
    logger.info(f"Markets to process: {', '.join(markets)}")
    logger.info("="*80)
    
    # Train model for each market
    for market in markets:
        logger.info(f"\n\nPROCESSING MARKET: {market}")
        
        try:
            config = MetaConfig(market=market)
            model, metrics = train_meta_model(config)
            all_metrics[market] = metrics
            logger.info(f"✅ {market} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {market} failed: {str(e)}")
            all_metrics[market] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    for market, metrics in all_metrics.items():
        if 'error' not in metrics:
            logger.info(f"{market}: Coverage={metrics.get('coverage', 0):.1%}, "
                       f"Abstention={metrics.get('abstention_rate', 0):.1%}, "
                       f"Accuracy={metrics.get('accuracy_when_trading', 0):.3f}")
        else:
            logger.info(f"{market}: Failed - {metrics['error'][:50]}...")
    
    logger.info("\n" + "="*80)
    logger.info("✅ All markets completed")
    logger.info("="*80)
    
    return all_metrics

if __name__ == "__main__":
    main()