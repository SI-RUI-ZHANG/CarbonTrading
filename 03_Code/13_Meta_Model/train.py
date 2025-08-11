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

from config import MetaConfig
from data_preparation import MetaDataPreparer
from model import create_meta_model

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
        preparer = MetaDataPreparer(config)
        X_train, y_train, X_val, y_val, _ = preparer.prepare_meta_training_data()
    
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
    
    # Create model based on configuration
    model = create_meta_model(config.META_MODEL_TYPE, config, input_size=X_train.shape[1])
    
    # Train model
    if config.META_MODEL_TYPE == 'neural_net':
        model.train_model(X_train, y_train, X_val, y_val)
    else:
        model.train(X_train, y_train, X_val, y_val)
    
    # ==================================================================================
    # STEP 3: EVALUATE PERFORMANCE
    # ==================================================================================
    
    logger.info("\n" + "-"*60)
    logger.info("STEP 3: EVALUATION")
    logger.info("-"*60)
    
    # Evaluate on training set
    train_metrics, train_pred, train_prob = model.evaluate(X_train, y_train, "Training")
    
    # Evaluate on validation set
    val_metrics, val_pred, val_prob = model.evaluate(X_val, y_val, "Validation")
    
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
    
    # Save model
    if config.META_MODEL_TYPE == 'xgboost':
        model.save(os.path.join(config.output_dir, 'meta_model.xgb'))
    elif config.META_MODEL_TYPE == 'logistic':
        model.save(os.path.join(config.output_dir, 'meta_model.pkl'))
    
    # Save metrics
    metrics = {
        'training': train_metrics,
        'validation': val_metrics,
        'config': {
            'model_type': config.META_MODEL_TYPE,
            'market': config.MARKET,
            'strategy': config.ENSEMBLE_STRATEGY,
            'confidence_threshold': config.CONFIDENCE_THRESHOLD
        }
    }
    
    with open(os.path.join(config.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    np.save(os.path.join(config.output_dir, 'val_predictions.npy'), val_pred)
    np.save(os.path.join(config.output_dir, 'val_probabilities.npy'), val_prob)
    
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
    Create visualization plots
    
    Args:
        model: Trained meta-model
        probabilities: Confidence scores
        true_correctness: Actual correctness
        config: MetaConfig object
    """
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


if __name__ == "__main__":
    # Train meta-model
    config = MetaConfig()
    model, metrics = train_meta_model(config)