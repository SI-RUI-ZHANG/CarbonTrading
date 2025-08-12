"""
Error Reversal Meta-Model
Identifies and reverses LSTM errors to improve overall accuracy.
No abstention - always makes predictions (100% coverage).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging
import joblib
import json
from typing import Tuple, Dict, Optional

# Try importing XGBoost, fall back to RandomForest if not available
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    USE_XGBOOST = False
    logging.warning("XGBoost not available, using RandomForest instead")

logger = logging.getLogger(__name__)


class ErrorReversalMetaModel:
    """
    Meta-model that identifies and reverses LSTM errors.
    Key difference from abstention model: ALWAYS makes predictions (100% coverage).
    """
    
    def __init__(self, config):
        """
        Initialize error reversal meta-model
        
        Args:
            config: MetaConfig object
        """
        self.config = config
        
        # Main model to predict if LSTM is WRONG
        if USE_XGBOOST:
            # Use parameters from config
            self.error_predictor = XGBClassifier(
                **config.ERROR_PREDICTOR_PARAMS,
                random_state=config.SEED,
                n_jobs=-1
            )
        else:
            # Fallback to RandomForest
            self.error_predictor = RandomForestClassifier(
                n_estimators=config.ERROR_PREDICTOR_PARAMS.get('n_estimators', 100),
                max_depth=config.ERROR_PREDICTOR_PARAMS.get('max_depth', 4),
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=config.SEED,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        # Threshold for reversing (will be optimized)
        self.reversal_threshold = 0.60
        
        # Track statistics
        self.reversal_stats = {}
        self.feature_importance = None
        
    def train(self, X_train, lstm_preds_train, y_train, 
              X_val=None, lstm_preds_val=None, y_val=None):
        """
        Train to identify LSTM errors
        
        Args:
            X_train: Training features
            lstm_preds_train: LSTM predictions on training set
            y_train: True labels for training set
            X_val, lstm_preds_val, y_val: Optional validation data
        """
        logger.info("Training Error Reversal Meta-Model...")
        
        # Target: 1 if LSTM was wrong, 0 if correct
        lstm_errors_train = (lstm_preds_train != y_train).astype(int)
        
        # Log error rate
        error_rate = lstm_errors_train.mean()
        logger.info(f"LSTM error rate on training: {error_rate:.1%}")
        
        # Train error predictor
        self.error_predictor.fit(X_train, lstm_errors_train)
        
        # Get feature importance
        if hasattr(self.error_predictor, 'feature_importances_'):
            self.feature_importance = self.error_predictor.feature_importances_
        
        # Optimize reversal threshold on validation set
        if X_val is not None and lstm_preds_val is not None and y_val is not None:
            self.optimize_threshold(X_val, lstm_preds_val, y_val)
        else:
            logger.info(f"Using default reversal threshold: {self.reversal_threshold}")
            
        logger.info("Error Reversal training completed")
        
    def optimize_threshold(self, X_val, lstm_preds_val, y_val):
        """
        Find optimal threshold for reversing predictions
        
        Args:
            X_val: Validation features
            lstm_preds_val: LSTM predictions on validation
            y_val: True validation labels
        """
        logger.info("Optimizing reversal threshold...")
        
        # Get error probabilities
        error_probs = self.error_predictor.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.4, 0.8, 0.05)
        best_accuracy = 0
        best_threshold = 0.6
        
        lstm_baseline = (lstm_preds_val == y_val).mean()
        
        for threshold in thresholds:
            # Apply reversals at this threshold
            should_reverse = error_probs > threshold
            final_preds = np.where(should_reverse, 1 - lstm_preds_val, lstm_preds_val)
            
            # Calculate accuracy
            accuracy = (final_preds == y_val).mean()
            reversal_rate = should_reverse.mean()
            
            # We want improvement but not too many reversals
            if accuracy > best_accuracy and reversal_rate < 0.3:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.reversal_threshold = best_threshold
        improvement = best_accuracy - lstm_baseline
        
        logger.info(f"Optimal threshold: {best_threshold:.2f}")
        logger.info(f"Validation accuracy: {lstm_baseline:.1%} → {best_accuracy:.1%} ({improvement:+.1%})")
        
    def predict_with_reversal(self, X, lstm_predictions):
        """
        Make predictions with selective reversal
        
        Args:
            X: Features
            lstm_predictions: Original LSTM predictions
            
        Returns:
            Tuple of (final_predictions, reversal_mask, error_probabilities)
        """
        # Predict probability that LSTM is wrong
        error_probs = self.error_predictor.predict_proba(X)[:, 1]
        
        # Decide which predictions to reverse
        should_reverse = error_probs > self.reversal_threshold
        
        # Apply reversals
        final_predictions = np.where(
            should_reverse,
            1 - lstm_predictions,  # Reverse
            lstm_predictions        # Keep
        )
        
        # Track statistics
        self.reversal_stats = {
            'total_samples': len(X),
            'reversed': should_reverse.sum(),
            'reversal_rate': should_reverse.mean(),
            'avg_error_prob': error_probs.mean(),
            'coverage': 1.0  # Always 100%
        }
        
        return final_predictions, should_reverse, error_probs
    
    def evaluate(self, X, lstm_preds, y, dataset_name="Test"):
        """
        Evaluate model with reversal metrics
        
        Args:
            X: Features
            lstm_preds: LSTM predictions
            y: True labels
            dataset_name: Name for logging
            
        Returns:
            Tuple of (metrics, final_predictions, error_probabilities)
        """
        # Get predictions with reversals
        final_preds, reversals, error_probs = self.predict_with_reversal(X, lstm_preds)
        
        # Calculate improvements
        lstm_accuracy = (lstm_preds == y).mean()
        final_accuracy = (final_preds == y).mean()
        improvement = final_accuracy - lstm_accuracy
        
        # Reversal statistics
        reversal_rate = reversals.mean()
        if reversals.any():
            # How many reversals were correct?
            reversal_success = (final_preds[reversals] == y[reversals]).mean()
            # How many non-reversals were correct?
            keep_success = (final_preds[~reversals] == y[~reversals]).mean() if (~reversals).any() else 0
        else:
            reversal_success = 0
            keep_success = lstm_accuracy
        
        # Confusion matrices
        lstm_cm = confusion_matrix(y, lstm_preds)
        final_cm = confusion_matrix(y, final_preds)
        
        # Calculate precision, recall, F1 for final predictions
        tn, fp, fn, tp = final_cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'lstm_baseline_accuracy': lstm_accuracy,
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'reversal_rate': reversal_rate,
            'reversals_correct': reversal_success,
            'kept_correct': keep_success,
            'coverage': 1.0,  # Always 100%
            'total_samples': len(y),
            'reversed_count': reversals.sum(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        # Log performance
        logger.info(f"\n{dataset_name} Set Performance:")
        logger.info(f"  Total samples: {len(y)}")
        logger.info(f"  Coverage: 100.0% (no abstention!)")
        logger.info(f"  Reversals: {reversals.sum()} ({reversal_rate:.1%})")
        
        logger.info(f"\nAccuracy Comparison:")
        logger.info(f"  LSTM Baseline: {lstm_accuracy:.1%}")
        logger.info(f"  After Reversals: {final_accuracy:.1%}")
        logger.info(f"  Improvement: {improvement:+.1%}")
        
        if reversals.any():
            logger.info(f"\nReversal Analysis:")
            logger.info(f"  Reversals success rate: {reversal_success:.1%}")
            logger.info(f"  Non-reversals success rate: {keep_success:.1%}")
        
        logger.info(f"\nConfusion Matrix (Final):")
        logger.info(f"  TN: {final_cm[0,0]:4d}  FP: {final_cm[0,1]:4d}")
        logger.info(f"  FN: {final_cm[1,0]:4d}  TP: {final_cm[1,1]:4d}")
        
        return metrics, final_preds, error_probs
    
    def save(self, filepath):
        """Save model to file"""
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata (convert numpy types)
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        metadata = {
            'reversal_threshold': float(self.reversal_threshold),
            'reversal_stats': convert_numpy(self.reversal_stats),
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'model_type': 'XGBoost' if USE_XGBOOST else 'RandomForest'
        }
        
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        return joblib.load(filepath)
    
    def plot_feature_importance(self, feature_names=None, save_path=None):
        """
        Plot feature importance for error detection
        
        Args:
            feature_names: List of feature names
            save_path: Path to save plot
        """
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
            
        import matplotlib.pyplot as plt
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_importance))]
        
        # Sort features by importance
        indices = np.argsort(self.feature_importance)[::-1][:10]  # Top 10
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), self.feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top Features for Error Detection')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()