"""
Ensemble Predictor that combines LSTM and Meta-Model predictions
Implements various strategies for using meta-model confidence
"""

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import joblib
import os
import logging
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Combines primary LSTM and meta-model for improved predictions
    """
    
    def __init__(self, config):
        """
        Initialize ensemble predictor
        
        Args:
            config: MetaConfig object
        """
        self.config = config
        self.primary_model = None
        self.meta_model = None
        self.scaler_primary = None
        self.scaler_meta = None
        self.is_loaded = False
        
    def load_models(self):
        """Load both primary LSTM and meta-model"""
        logger.info("Loading ensemble models...")
        
        # Load primary LSTM model
        primary_model_path = os.path.join(self.config.PRIMARY_MODEL_DIR, 'best_model.pth')
        if os.path.exists(primary_model_path):
            # Load model checkpoint
            checkpoint = torch.load(primary_model_path, map_location='cpu')
            logger.info(f"Loaded primary LSTM from {primary_model_path}")
        else:
            logger.warning(f"Primary model not found at {primary_model_path}")
        
        # Load primary scaler
        primary_scaler_path = os.path.join(
            '../../02_Data_Processed/04_LSTM_Ready',
            f'{self.config.MARKET}_scaler.pkl'
        )
        if os.path.exists(primary_scaler_path):
            self.scaler_primary = joblib.load(primary_scaler_path)
        
        # Load meta-model
        meta_model_path = os.path.join(self.config.output_dir, 'meta_model.xgb')
        if os.path.exists(meta_model_path):
            self.meta_model = xgb.Booster()
            self.meta_model.load_model(meta_model_path)
            logger.info(f"Loaded meta-model from {meta_model_path}")
        else:
            logger.warning(f"Meta-model not found at {meta_model_path}")
        
        # Load meta scaler
        meta_scaler_path = os.path.join(self.config.output_dir, 'meta_scaler.pkl')
        if os.path.exists(meta_scaler_path):
            self.scaler_meta = joblib.load(meta_scaler_path)
        
        self.is_loaded = True
        
    def predict_with_ensemble(
        self, 
        primary_features: np.ndarray,
        sentiment_features: np.ndarray,
        strategy: Optional[str] = None
    ) -> Dict:
        """
        Make predictions using ensemble strategy
        
        Args:
            primary_features: Features for primary LSTM (shape: [batch, seq_len, features])
            sentiment_features: Features for meta-model (shape: [batch, features])
            strategy: Override config strategy if provided
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.is_loaded:
            self.load_models()
        
        strategy = strategy or self.config.ENSEMBLE_STRATEGY
        
        # Get primary LSTM predictions
        # Note: In real implementation, would need to properly load and run LSTM
        # For now, simulate with random predictions
        lstm_predictions = np.random.randint(0, 2, len(primary_features))
        lstm_probabilities = np.random.rand(len(primary_features))
        
        # Prepare features for meta-model
        # Add LSTM prediction and day of week as additional features
        batch_size = len(sentiment_features)
        additional_features = np.column_stack([
            lstm_predictions,
            np.zeros(batch_size)  # Placeholder for day of week
        ])
        
        meta_features = np.hstack([sentiment_features, additional_features])
        
        # Scale features
        if self.scaler_meta is not None:
            meta_features = self.scaler_meta.transform(meta_features)
        
        # Get meta-model confidence scores
        dmeta = xgb.DMatrix(meta_features)
        meta_confidence = self.meta_model.predict(dmeta) if self.meta_model else np.ones(batch_size) * 0.5
        
        # Apply ensemble strategy
        results = self._apply_strategy(
            lstm_predictions,
            lstm_probabilities,
            meta_confidence,
            strategy
        )
        
        return results
    
    def _apply_strategy(
        self,
        lstm_predictions: np.ndarray,
        lstm_probabilities: np.ndarray,
        meta_confidence: np.ndarray,
        strategy: str,
        volatility: np.ndarray = None,
        sentiment_scores: np.ndarray = None
    ) -> Dict:
        """
        Apply ensemble strategy to combine predictions
        
        Args:
            lstm_predictions: Binary predictions from LSTM
            lstm_probabilities: Confidence scores from LSTM
            meta_confidence: Confidence scores from meta-model
            strategy: Strategy name
            volatility: Rolling volatility for volatility_gated strategy
            sentiment_scores: Raw sentiment scores for contrarian signals
            
        Returns:
            Results dictionary
        """
        batch_size = len(lstm_predictions)
        
        if strategy == 'filtered':
            # Only trade when meta-model is confident
            trade_mask = meta_confidence > self.config.CONFIDENCE_THRESHOLD
            final_predictions = lstm_predictions.copy()
            final_predictions[~trade_mask] = -1  # -1 means "don't trade"
            
        elif strategy == 'weighted':
            # Use meta confidence for position sizing
            final_predictions = lstm_predictions.copy()
            position_sizes = self._calculate_position_sizes(meta_confidence)
            
        elif strategy == 'selective':
            # Follow LSTM only when meta says it's reliable
            reliable_mask = meta_confidence > 0.5
            final_predictions = np.full(batch_size, -1)  # Default to no trade
            final_predictions[reliable_mask] = lstm_predictions[reliable_mask]
            
        elif strategy == 'contrarian':
            # When meta strongly disagrees, consider inverting
            low_conf_mask = meta_confidence < 0.3
            final_predictions = lstm_predictions.copy()
            final_predictions[low_conf_mask] = 1 - final_predictions[low_conf_mask]
            
        elif strategy == 'volatility_gated':
            # Trade only in low volatility with high sentiment confidence
            if volatility is not None:
                # Calculate volatility percentile
                from scipy import stats
                vol_percentile = stats.rankdata(volatility, method='average') / len(volatility)
                
                # Gate conditions
                low_vol_mask = vol_percentile < 0.2  # Bottom 20% volatility
                high_confidence = meta_confidence > 0.6
                
                # Contrarian on extreme negative sentiment
                if sentiment_scores is not None:
                    sentiment_percentile = stats.rankdata(sentiment_scores, method='average') / len(sentiment_scores)
                    extreme_negative = sentiment_percentile < 0.1
                    
                    # Apply strategy
                    final_predictions = np.full(batch_size, -1)  # Default no trade
                    
                    # Normal trades in low vol + high confidence
                    normal_trade = low_vol_mask & high_confidence & ~extreme_negative
                    final_predictions[normal_trade] = lstm_predictions[normal_trade]
                    
                    # Contrarian trades on extreme negative sentiment in low vol
                    contrarian_trade = low_vol_mask & extreme_negative
                    final_predictions[contrarian_trade] = 1 - lstm_predictions[contrarian_trade]
                else:
                    # Without sentiment scores, just use volatility gating
                    trade_mask = low_vol_mask & high_confidence
                    final_predictions = np.full(batch_size, -1)
                    final_predictions[trade_mask] = lstm_predictions[trade_mask]
            else:
                # Fallback to filtered if no volatility data
                trade_mask = meta_confidence > self.config.CONFIDENCE_THRESHOLD
                final_predictions = lstm_predictions.copy()
                final_predictions[~trade_mask] = -1
            
        else:  # 'baseline' or unknown
            final_predictions = lstm_predictions
            
        # Calculate statistics
        trade_count = np.sum(final_predictions != -1)
        coverage = trade_count / batch_size if batch_size > 0 else 0
        
        results = {
            'predictions': final_predictions,
            'lstm_predictions': lstm_predictions,
            'lstm_probabilities': lstm_probabilities,
            'meta_confidence': meta_confidence,
            'coverage': coverage,
            'trade_count': trade_count,
            'strategy': strategy
        }
        
        if strategy == 'weighted':
            results['position_sizes'] = position_sizes
            
        return results
    
    def _calculate_position_sizes(self, confidence: np.ndarray) -> np.ndarray:
        """
        Calculate position sizes based on confidence
        
        Args:
            confidence: Meta-model confidence scores
            
        Returns:
            Position sizes (0 to 1)
        """
        # Linear scaling from min to max position size
        min_size = self.config.MIN_POSITION_SIZE
        max_size = self.config.MAX_POSITION_SIZE
        
        # Clip confidence to [0, 1]
        confidence = np.clip(confidence, 0, 1)
        
        # Linear interpolation
        position_sizes = min_size + (max_size - min_size) * confidence
        
        # Don't trade if confidence is too low
        position_sizes[confidence < 0.3] = 0
        
        return position_sizes
    
    def evaluate_strategies(
        self,
        test_features: np.ndarray,
        test_sentiment: np.ndarray,
        test_targets: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate multiple ensemble strategies
        
        Args:
            test_features: Test features for LSTM
            test_sentiment: Test sentiment features
            test_targets: True labels
            
        Returns:
            DataFrame with strategy comparison
        """
        strategies = ['baseline', 'filtered', 'selective', 'weighted', 'contrarian']
        results = []
        
        for strategy in strategies:
            preds = self.predict_with_ensemble(
                test_features,
                test_sentiment,
                strategy=strategy
            )
            
            # Calculate metrics
            mask = preds['predictions'] != -1  # Trades made
            if mask.sum() > 0:
                accuracy = (preds['predictions'][mask] == test_targets[mask]).mean()
            else:
                accuracy = 0
            
            results.append({
                'strategy': strategy,
                'coverage': preds['coverage'],
                'trades': preds['trade_count'],
                'accuracy': accuracy
            })
        
        return pd.DataFrame(results)


def visualize_ensemble_results(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize ensemble strategy comparison
    
    Args:
        results_df: DataFrame with strategy results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Coverage comparison
    axes[0].bar(results_df['strategy'], results_df['coverage'], color='steelblue')
    axes[0].set_xlabel('Strategy')
    axes[0].set_ylabel('Coverage')
    axes[0].set_title('Trading Coverage by Strategy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    axes[1].bar(results_df['strategy'], results_df['accuracy'], color='green')
    axes[1].set_xlabel('Strategy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Strategy')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Coverage vs Accuracy scatter
    axes[2].scatter(results_df['coverage'], results_df['accuracy'], s=100, alpha=0.6)
    for idx, row in results_df.iterrows():
        axes[2].annotate(row['strategy'], 
                        (row['coverage'], row['accuracy']),
                        xytext=(5, 5), textcoords='offset points')
    axes[2].set_xlabel('Coverage')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Coverage vs Accuracy Trade-off')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def main():
    """Demo ensemble predictor"""
    from config import MetaConfig
    
    config = MetaConfig()
    ensemble = EnsemblePredictor(config)
    
    # Load models
    ensemble.load_models()
    
    # Simulate some test data
    batch_size = 100
    test_features = np.random.randn(batch_size, 60, 49)  # LSTM features
    test_sentiment = np.random.randn(batch_size, 10)  # Sentiment features
    test_targets = np.random.randint(0, 2, batch_size)
    
    # Evaluate strategies
    results = ensemble.evaluate_strategies(test_features, test_sentiment, test_targets)
    
    print("\nStrategy Comparison:")
    print(results.to_string(index=False))
    
    # Visualize
    visualize_ensemble_results(results, 
                              save_path=os.path.join(config.output_dir, 'strategy_comparison.png'))


if __name__ == "__main__":
    main()