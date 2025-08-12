"""
Random Forest Meta-Model with Abstention Mechanism
Predicts LSTM reliability using sentiment features with focus on identifying when NOT to trade
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class RandomForestMetaModel:
    """
    Random Forest meta-model with abstention mechanism for sparse sentiment data.
    Focuses on identifying when NOT to trade rather than forcing predictions.
    """
    
    def __init__(self, config):
        """
        Initialize Random Forest meta-model with abstention capabilities
        
        Args:
            config: MetaConfig object
        """
        self.config = config
        self.model = None
        self.feature_importance = None
        self.oob_score = None
        self.abstention_stats = {}
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Random Forest model with out-of-bag scoring
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional, used for monitoring)
        """
        logger.info("Training Random Forest meta-model with abstention mechanism...")
        
        # Calculate feature presence statistics
        feature_coverage = self._calculate_feature_coverage(X_train)
        logger.info(f"Training data feature coverage: {feature_coverage:.1%}")
        
        # Initialize Random Forest with conservative parameters
        self.model = RandomForestClassifier(
            n_estimators=self.config.RANDOM_FOREST_PARAMS['n_estimators'],
            max_depth=self.config.RANDOM_FOREST_PARAMS['max_depth'],
            min_samples_leaf=self.config.RANDOM_FOREST_PARAMS['min_samples_leaf'],
            min_samples_split=self.config.RANDOM_FOREST_PARAMS.get('min_samples_split', 20),
            max_features=self.config.RANDOM_FOREST_PARAMS['max_features'],
            bootstrap=self.config.RANDOM_FOREST_PARAMS['bootstrap'],
            oob_score=True,  # Enable out-of-bag scoring
            random_state=self.config.SEED,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Get OOB score (honest estimate of performance)
        self.oob_score = self.model.oob_score_
        logger.info(f"Out-of-bag score: {self.oob_score:.3f}")
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Calculate tree agreement statistics
        self._analyze_tree_agreement(X_train)
        
        # Validation performance if provided
        if X_val is not None and y_val is not None:
            val_coverage = self._calculate_feature_coverage(X_val)
            logger.info(f"Validation data feature coverage: {val_coverage:.1%}")
            
            # Get predictions with abstention
            predictions, probabilities, abstentions = self.predict_with_abstention(X_val)
            
            # Calculate metrics only on non-abstained samples
            traded_mask = ~abstentions
            if traded_mask.sum() > 0:
                val_accuracy = accuracy_score(y_val[traded_mask], predictions[traded_mask])
                logger.info(f"Validation accuracy (when trading): {val_accuracy:.3f}")
                logger.info(f"Abstention rate: {abstentions.mean():.1%}")
            else:
                logger.warning("All validation samples abstained!")
        
        logger.info("Random Forest training completed")
        
    def predict_with_abstention(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with abstention based on confidence and feature presence
        
        Args:
            X: Input features
            
        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Confidence scores
            abstentions: Boolean mask of abstained predictions
        """
        # Calculate feature coverage for each sample
        feature_presence = self._calculate_sample_feature_presence(X)
        
        # Get tree predictions for uncertainty estimation
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate tree agreement (variance across trees)
        tree_agreement = tree_predictions.std(axis=0)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        # Abstention logic
        abstentions = self._should_abstain(
            feature_presence,
            tree_agreement,
            probabilities
        )
        
        # Log abstention statistics
        self.abstention_stats = {
            'total_samples': len(X),
            'abstained': abstentions.sum(),
            'abstention_rate': abstentions.mean(),
            'low_feature_coverage': (feature_presence < self.config.MIN_SENTIMENT_COVERAGE).sum(),
            'high_tree_disagreement': (tree_agreement > self.config.ABSTENTION_THRESHOLD).sum(),
            'low_confidence': ((probabilities < 0.4) | (probabilities > 0.6)).sum()
        }
        
        return predictions, probabilities, abstentions
    
    def _should_abstain(self, feature_presence, tree_agreement, probabilities) -> np.ndarray:
        """
        Determine whether to abstain from trading for each sample
        
        Args:
            feature_presence: Fraction of features present per sample
            tree_agreement: Standard deviation of tree predictions
            probabilities: Model confidence scores
            
        Returns:
            Boolean mask where True = abstain from trading
        """
        abstain = np.zeros(len(feature_presence), dtype=bool)
        
        # Abstain if insufficient features
        abstain |= (feature_presence < self.config.MIN_SENTIMENT_COVERAGE)
        
        # Abstain if trees disagree too much
        abstain |= (tree_agreement > self.config.ABSTENTION_THRESHOLD)
        
        # Abstain if confidence is not strong enough (too close to 50%)
        # We want to trade only when confidence is < MIN or > MAX (far from 50%)
        # So abstain when confidence is between MIN and MAX
        abstain |= (probabilities >= self.config.MIN_CONFIDENCE_TO_TRADE) & \
                   (probabilities <= self.config.MAX_CONFIDENCE_TO_TRADE)
        
        return abstain
    
    def _calculate_feature_coverage(self, X) -> float:
        """Calculate overall feature coverage in dataset"""
        # Assuming 0 or NaN indicates missing feature
        non_zero_mask = (X != 0) & ~np.isnan(X)
        return non_zero_mask.mean()
    
    def _calculate_sample_feature_presence(self, X) -> np.ndarray:
        """Calculate feature presence for each sample"""
        # Count non-zero, non-NaN features per sample
        non_zero_mask = (X != 0) & ~np.isnan(X)
        return non_zero_mask.mean(axis=1)
    
    def _analyze_tree_agreement(self, X_sample):
        """Analyze tree agreement patterns on training data"""
        # Take a sample for analysis (to save computation)
        sample_size = min(100, len(X_sample))
        sample_idx = np.random.choice(len(X_sample), sample_size, replace=False)
        X_analysis = X_sample[sample_idx]
        
        # Get individual tree predictions
        tree_predictions = np.array([tree.predict(X_analysis) for tree in self.model.estimators_])
        
        # Calculate agreement statistics
        agreement_mean = tree_predictions.mean(axis=0)
        agreement_std = tree_predictions.std(axis=0)
        
        logger.info(f"Tree agreement analysis:")
        logger.info(f"  Mean agreement: {agreement_mean.mean():.3f}")
        logger.info(f"  Mean disagreement (std): {agreement_std.mean():.3f}")
        logger.info(f"  Max disagreement: {agreement_std.max():.3f}")
        
    def evaluate(self, X, y, dataset_name="Test"):
        """
        Evaluate model performance with abstention metrics
        
        Args:
            X, y: Data to evaluate
            dataset_name: Name for logging
        """
        predictions, probabilities, abstentions = self.predict_with_abstention(X)
        
        # Calculate metrics only on traded samples
        traded_mask = ~abstentions
        traded_count = traded_mask.sum()
        
        logger.info(f"\n{dataset_name} Set Performance:")
        logger.info(f"  Total samples: {len(y)}")
        logger.info(f"  Abstained: {abstentions.sum()} ({abstentions.mean():.1%})")
        logger.info(f"  Traded: {traded_count} ({traded_mask.mean():.1%})")
        
        if traded_count > 0:
            metrics = {
                'accuracy': accuracy_score(y[traded_mask], predictions[traded_mask]),
                'precision': precision_score(y[traded_mask], predictions[traded_mask], zero_division=0),
                'recall': recall_score(y[traded_mask], predictions[traded_mask]),
                'f1_score': f1_score(y[traded_mask], predictions[traded_mask]),
                'coverage': traded_mask.mean(),
                'abstention_rate': abstentions.mean()
            }
            
            # Add AUC if we have both classes
            if len(np.unique(y[traded_mask])) > 1:
                metrics['auc_roc'] = roc_auc_score(y[traded_mask], probabilities[traded_mask])
            else:
                metrics['auc_roc'] = 0
            
            logger.info("\nMetrics (on traded samples only):")
            for metric, value in metrics.items():
                logger.info(f"  {metric:15s}: {value:.3f}")
            
            # Confusion matrix for traded samples
            cm = confusion_matrix(y[traded_mask], predictions[traded_mask])
            logger.info(f"\nConfusion Matrix (traded only):")
            if cm.shape == (2, 2):
                logger.info(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
                logger.info(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
            else:
                logger.info(f"  Matrix shape {cm.shape}: {cm}")
            
            # Abstention analysis
            logger.info(f"\nAbstention Analysis:")
            for key, value in self.abstention_stats.items():
                logger.info(f"  {key}: {value}")
            
        else:
            logger.warning("All samples abstained - no metrics to calculate")
            metrics = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 
                'f1_score': 0, 'auc_roc': 0, 
                'coverage': 0, 'abstention_rate': 1.0
            }
        
        return metrics, predictions, probabilities, abstentions
    
    def plot_feature_importance(self, feature_names=None, save_path=None):
        """Plot feature importance with focus on reliable features"""
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        # Use provided names or generate defaults
        if feature_names is None:
            feature_names = self.config.SENTIMENT_FEATURES + ['lstm_pred', 'day_of_week']
            feature_names = feature_names[:len(self.feature_importance)]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(self.feature_importance)],
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top 10 features
        top_features = importance_df.head(10)
        ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 10 Features - Random Forest Meta-Model')
        ax1.invert_yaxis()
        
        # Feature importance distribution
        ax2.hist(self.feature_importance, bins=20, color='darkgreen', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Importance Distribution')
        ax2.axvline(self.feature_importance.mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.feature_importance.mean():.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
    def plot_abstention_analysis(self, X, y, save_path=None):
        """Visualize abstention patterns and their impact"""
        predictions, probabilities, abstentions = self.predict_with_abstention(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence distribution for abstained vs traded
        ax = axes[0, 0]
        ax.hist(probabilities[abstentions], bins=20, alpha=0.5, label='Abstained', color='red')
        ax.hist(probabilities[~abstentions], bins=20, alpha=0.5, label='Traded', color='green')
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution by Trading Decision')
        ax.legend()
        
        # 2. Feature coverage impact
        ax = axes[0, 1]
        feature_presence = self._calculate_sample_feature_presence(X)
        ax.scatter(feature_presence[~abstentions], probabilities[~abstentions], 
                  alpha=0.5, label='Traded', color='green')
        ax.scatter(feature_presence[abstentions], probabilities[abstentions], 
                  alpha=0.5, label='Abstained', color='red')
        ax.set_xlabel('Feature Coverage')
        ax.set_ylabel('Prediction Confidence')
        ax.set_title('Feature Coverage vs Confidence')
        ax.axvline(self.config.MIN_SENTIMENT_COVERAGE, color='black', linestyle='--', 
                  label='Min Coverage Threshold')
        ax.legend()
        
        # 3. Accuracy by confidence buckets (traded only)
        ax = axes[1, 0]
        traded_mask = ~abstentions
        if traded_mask.sum() > 0:
            confidence_buckets = pd.cut(probabilities[traded_mask], bins=5)
            accuracy_by_bucket = pd.DataFrame({
                'confidence': confidence_buckets,
                'correct': (predictions[traded_mask] == y[traded_mask])
            }).groupby('confidence')['correct'].mean()
            
            accuracy_by_bucket.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Confidence Bucket')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Confidence Level (Traded Only)')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        # 4. Abstention reasons breakdown
        ax = axes[1, 1]
        feature_presence = self._calculate_sample_feature_presence(X)
        reasons = {
            'Low Feature\nCoverage': (feature_presence < self.config.MIN_SENTIMENT_COVERAGE).sum(),
            'Low\nConfidence': ((probabilities > 0.4) & (probabilities < 0.6)).sum(),
            'High Tree\nDisagreement': 0  # Would need tree predictions
        }
        ax.bar(reasons.keys(), reasons.values(), color='coral')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Abstention Reasons')
        
        plt.suptitle(f'Abstention Analysis - Coverage: {(~abstentions).mean():.1%}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
    def save(self, path):
        """Save model and metadata"""
        # Save model
        joblib.dump(self.model, path)
        
        # Save additional metadata (convert numpy types to Python types)
        metadata = {
            'oob_score': float(self.oob_score) if self.oob_score is not None else None,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'abstention_stats': {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v 
                                for k, v in self.abstention_stats.items()},
            'n_estimators': int(self.model.n_estimators),
            'max_depth': int(self.model.max_depth) if self.model.max_depth is not None else None
        }
        
        metadata_path = path.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, path):
        """Load model and metadata"""
        self.model = joblib.load(path)
        
        # Load metadata if available
        metadata_path = path.replace('.pkl', '_metadata.json').replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.oob_score = metadata.get('oob_score')
                self.feature_importance = np.array(metadata.get('feature_importance', []))
                self.abstention_stats = metadata.get('abstention_stats', {})
        
        logger.info(f"Model loaded from {path}")


def create_meta_model(model_type, config, input_size=None):
    """
    Factory function to create meta-model
    Now only supports Random Forest with abstention
    
    Args:
        model_type: Should be 'random_forest'
        config: MetaConfig object
        input_size: Not used for Random Forest
    
    Returns:
        RandomForestMetaModel instance
    """
    if model_type != 'random_forest':
        logger.warning(f"Model type '{model_type}' not supported. Using Random Forest.")
    
    return RandomForestMetaModel(config)