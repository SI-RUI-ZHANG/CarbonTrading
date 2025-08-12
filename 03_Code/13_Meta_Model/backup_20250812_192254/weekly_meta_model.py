"""
Weekly Meta Model with Abstention Mechanism
Random Forest model that predicts LSTM reliability using sentiment features
Includes abstention when confidence is low
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple, Dict, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeeklyMetaModel:
    """
    Meta model that predicts when weekly LSTM will be correct
    Uses sentiment features to assess LSTM reliability
    Includes abstention mechanism for low confidence scenarios
    """
    
    def __init__(self, config):
        """
        Initialize meta model
        
        Args:
            config: WeeklyConfig object
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.oob_score = None
        self.abstention_stats = {}
        
    def create_meta_features(self, sentiment_df: pd.DataFrame, 
                            lstm_predictions: np.ndarray,
                            lstm_probabilities: np.ndarray) -> np.ndarray:
        """
        Create features for meta model
        
        Args:
            sentiment_df: DataFrame with sentiment features
            lstm_predictions: Binary predictions from LSTM
            lstm_probabilities: Probability predictions from LSTM
            
        Returns:
            Feature array for meta model
        """
        # Sentiment features
        sentiment_features = sentiment_df.values
        
        # LSTM confidence features
        lstm_confidence = np.abs(lstm_probabilities - 0.5) * 2  # Scale to [0, 1]
        
        # LSTM prediction
        lstm_pred = lstm_predictions.reshape(-1, 1)
        
        # LSTM probability
        lstm_prob = lstm_probabilities.reshape(-1, 1)
        
        # Day of week (cyclical encoding)
        if isinstance(sentiment_df.index, pd.DatetimeIndex):
            dow = sentiment_df.index.dayofweek.values
            dow_sin = np.sin(2 * np.pi * dow / 7).reshape(-1, 1)
            dow_cos = np.cos(2 * np.pi * dow / 7).reshape(-1, 1)
        else:
            dow_sin = np.zeros((len(sentiment_df), 1))
            dow_cos = np.zeros((len(sentiment_df), 1))
        
        # Combine all features
        meta_features = np.hstack([
            sentiment_features,
            lstm_pred,
            lstm_prob,
            lstm_confidence.reshape(-1, 1),
            dow_sin,
            dow_cos
        ])
        
        return meta_features
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train Random Forest meta model
        
        Args:
            X_train: Training features
            y_train: Training labels (1 if LSTM correct, 0 if wrong)
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        logger.info("Training Weekly Meta Model (Random Forest)")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize Random Forest with weekly-tuned parameters
        self.model = RandomForestClassifier(**self.config.RF_PARAMS)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Get OOB score if available
        if self.config.RF_PARAMS.get('oob_score', False):
            self.oob_score = self.model.oob_score_
            logger.info(f"OOB Score: {self.oob_score:.3f}")
        
        # Feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Training accuracy
        train_pred = self.model.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)
        logger.info(f"Training Accuracy: {train_acc:.3f}")
        
        # Validation accuracy if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            val_acc = np.mean(val_pred == y_val)
            logger.info(f"Validation Accuracy: {val_acc:.3f}")
            
            # Get validation probabilities
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            logger.info(f"Validation probability range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")
    
    def predict_with_abstention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with abstention mechanism
        
        Args:
            X: Feature array
            
        Returns:
            Tuple of (predictions, probabilities, abstentions)
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Calculate feature coverage for each sample
        feature_presence = self._calculate_sample_feature_presence(X)
        
        # Get individual tree predictions for uncertainty
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        
        # Calculate tree agreement (std across trees)
        tree_agreement = tree_predictions.std(axis=0)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        # Abstention logic (adjusted for weekly data)
        abstentions = self._should_abstain_weekly(
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
            'low_confidence': self._count_low_confidence(probabilities)
        }
        
        logger.info(f"Abstention rate: {abstentions.mean():.1%} ({abstentions.sum()}/{len(X)} samples)")
        
        return predictions, probabilities, abstentions
    
    def _calculate_sample_feature_presence(self, X: np.ndarray) -> np.ndarray:
        """Calculate feature presence for each sample"""
        # For weekly data, consider non-zero values as present
        # Weekly aggregation should have better coverage
        non_zero_mask = (X != 0) & ~np.isnan(X)
        
        # Focus on sentiment features (first N features)
        n_sentiment = len(self.config.SENTIMENT_FEATURES)
        if X.shape[1] > n_sentiment:
            # Only check sentiment feature coverage
            sentiment_coverage = non_zero_mask[:, :n_sentiment].mean(axis=1)
        else:
            sentiment_coverage = non_zero_mask.mean(axis=1)
        
        return sentiment_coverage
    
    def _should_abstain_weekly(self, feature_presence: np.ndarray,
                              tree_agreement: np.ndarray,
                              probabilities: np.ndarray) -> np.ndarray:
        """
        Determine abstention for weekly predictions
        Adjusted thresholds for weekly aggregated data
        
        Args:
            feature_presence: Fraction of features present per sample
            tree_agreement: Standard deviation of tree predictions
            probabilities: Model confidence scores
            
        Returns:
            Boolean mask where True = abstain from trading
        """
        abstain = np.zeros(len(feature_presence), dtype=bool)
        
        # Abstain if insufficient sentiment coverage (adjusted for weekly)
        abstain |= (feature_presence < self.config.MIN_SENTIMENT_COVERAGE)
        
        # Abstain if trees disagree too much (adjusted for weekly)
        abstain |= (tree_agreement > self.config.ABSTENTION_THRESHOLD)
        
        # Abstain if confidence is too close to 50% (adjusted for weekly)
        confidence_margin = np.abs(probabilities - 0.5)
        abstain |= (confidence_margin < self.config.CONFIDENCE_MARGIN)
        
        # Additional weekly-specific rule: abstain if extreme probability but low coverage
        extreme_prob = (probabilities < 0.2) | (probabilities > 0.8)
        low_coverage = feature_presence < 0.1
        abstain |= (extreme_prob & low_coverage)
        
        return abstain
    
    def _count_low_confidence(self, probabilities: np.ndarray) -> int:
        """Count samples with low confidence"""
        confidence_margin = np.abs(probabilities - 0.5)
        return (confidence_margin < self.config.CONFIDENCE_MARGIN).sum()
    
    def evaluate_abstention_strategy(self, X: np.ndarray, y_true: np.ndarray,
                                    lstm_predictions: np.ndarray) -> Dict:
        """
        Evaluate the abstention strategy
        
        Args:
            X: Feature array
            y_true: True labels
            lstm_predictions: LSTM predictions to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get meta model predictions with abstention
        meta_pred, meta_prob, abstentions = self.predict_with_abstention(X)
        
        # Calculate metrics on traded samples
        traded_mask = ~abstentions
        coverage = traded_mask.mean()
        
        if traded_mask.sum() > 0:
            # Accuracy when trading
            traded_predictions = lstm_predictions[traded_mask]
            traded_actuals = y_true[traded_mask]
            accuracy_when_trading = np.mean(traded_predictions == traded_actuals)
            
            # Compare to baseline (no abstention)
            baseline_accuracy = np.mean(lstm_predictions == y_true)
            improvement = accuracy_when_trading - baseline_accuracy
        else:
            accuracy_when_trading = 0
            baseline_accuracy = np.mean(lstm_predictions == y_true)
            improvement = 0
        
        metrics = {
            'coverage': coverage,
            'abstention_rate': 1 - coverage,
            'trades_made': traded_mask.sum(),
            'total_samples': len(y_true),
            'accuracy_when_trading': accuracy_when_trading,
            'baseline_accuracy': baseline_accuracy,
            'improvement': improvement,
            'mean_confidence': meta_prob.mean(),
            'std_confidence': meta_prob.std()
        }
        
        return metrics
    
    def save(self, output_dir: str):
        """Save meta model and related files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'weekly_meta_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'weekly_meta_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'oob_score': self.oob_score,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None,
            'abstention_stats': self.abstention_stats,
            'config': {
                'market': self.config.MARKET,
                'min_sentiment_coverage': self.config.MIN_SENTIMENT_COVERAGE,
                'abstention_threshold': self.config.ABSTENTION_THRESHOLD,
                'confidence_margin': self.config.CONFIDENCE_MARGIN
            }
        }
        
        import json
        metadata_path = os.path.join(output_dir, 'weekly_meta_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved meta model to {output_dir}")
    
    def load(self, output_dir: str):
        """Load meta model and related files"""
        # Load model
        model_path = os.path.join(output_dir, 'weekly_meta_model.pkl')
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(output_dir, 'weekly_meta_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        import json
        metadata_path = os.path.join(output_dir, 'weekly_meta_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.oob_score = metadata.get('oob_score')
                self.feature_importance = np.array(metadata.get('feature_importance', []))
                self.abstention_stats = metadata.get('abstention_stats', {})
        
        logger.info(f"Loaded meta model from {output_dir}")