"""
Configuration for Error Reversal Meta-Model
The meta-model identifies LSTM errors and reverses them to improve accuracy.
No abstention - always makes predictions (100% coverage).
"""

import os
from datetime import datetime


class ReversalConfig:
    """Configuration for error reversal meta-model training and evaluation"""
    
    def __init__(self, market='GDEA', frequency='daily'):
        """
        Initialize config with specific market and frequency
        
        Args:
            market: 'GDEA' or 'HBEA'
            frequency: 'daily' or 'weekly'
        """
        self.MARKET = market
        self.FREQUENCY = frequency
    
    # ==================================================================================
    # EXPERIMENT CONFIGURATION
    # ==================================================================================
    
    # Primary LSTM model to evaluate
    @property
    def PRIMARY_MODEL_DIR(self):
        """Path to primary LSTM model (base model without sentiment)"""
        return f'../../04_Models/{self.FREQUENCY}_{self.MARKET}_base'
    
    # ==================================================================================
    # DATA PATHS
    # ==================================================================================
    
    # Prepared LSTM sequences (to get exact split indices)
    LSTM_DATA_DIR = '../../02_Data_Processed/04_LSTM_Ready/'
    
    # Fixed output directory structure (no timestamps)
    OUTPUT_BASE_DIR = '../../04_Models/'
    
    @property
    def SENTIMENT_DATA_PATH(self):
        """Data with sentiment features"""
        if self.FREQUENCY == 'daily':
            return f'../../02_Data_Processed/10_Sentiment_Final_Merged/{self.MARKET}_LSTM_with_sentiment.parquet'
        else:  # weekly
            return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly_with_sentiment.parquet'
    
    @property
    def BASE_DATA_PATH(self):
        """Data without sentiment (for primary LSTM)"""
        if self.FREQUENCY == 'daily':
            return f'../../02_Data_Processed/03_Feature_Engineered/{self.MARKET}_LSTM_advanced.parquet'
        else:  # weekly
            return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly.parquet'
    
    # ==================================================================================
    # ERROR DETECTION FEATURES
    # ==================================================================================
    
    # All features used for error detection (sentiment + LSTM + market)
    SENTIMENT_FEATURES = [
        'sentiment_supply',
        'sentiment_demand', 
        'supply_decayed',
        'demand_decayed',
        'policy_decayed',
        'supply_momentum',
        'demand_momentum',
        'doc_count',
        'max_policy',
        'avg_policy'
    ]
    
    LSTM_FEATURES = [
        'lstm_prediction',      # What LSTM predicted (0/1)
        'lstm_probability',     # LSTM confidence (0-1)
        'lstm_confidence',      # Distance from 0.5
        'lstm_entropy'          # Uncertainty measure
    ]
    
    ERROR_DETECTION_FEATURES = [
        'lstm_uncertainty',              # Near 50% confidence
        'lstm_extreme_conf',             # Very high or low confidence
        'sentiment_agrees_with_lstm',    # Agreement indicator
        'sentiment_strong_disagree',     # Strong disagreement
        'confidence_sentiment_interaction'  # Interaction term
    ]
    
    # ==================================================================================
    # ERROR REVERSAL CONFIGURATION
    # ==================================================================================
    
    # Model type for error detection
    MODEL_TYPE = 'error_reversal'  # New model type
    USE_XGBOOST = True  # Try to use XGBoost if available
    
    # Error predictor parameters (XGBoost)
    ERROR_PREDICTOR_PARAMS = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1.2  # Slightly favor finding errors
    }
    
    # Reversal thresholds
    INITIAL_REVERSAL_THRESHOLD = 0.60  # Starting threshold
    MIN_REVERSAL_CONFIDENCE = 0.55     # Minimum confidence to reverse
    MAX_REVERSAL_RATE = 0.30           # Don't reverse more than 30%
    OPTIMAL_REVERSAL_RATE = 0.15       # Target ~15% reversals
    
    # NO ABSTENTION - Always make predictions
    ALWAYS_TRADE = True
    COVERAGE = 1.0  # Always 100%
    
    # ==================================================================================
    # TRAINING CONFIGURATION
    # ==================================================================================
    
    # Use same splits as primary LSTM
    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Sequence length (must match primary LSTM)
    @property
    def SEQUENCE_LENGTH(self):
        return 60 if self.FREQUENCY == 'daily' else 12  # 60 days or 12 weeks
    
    # Random seed for reproducibility
    SEED = 42
    
    # ==================================================================================
    # EVALUATION METRICS
    # ==================================================================================
    
    # Metrics to track for error reversal
    METRICS = [
        'lstm_baseline_accuracy',   # Original LSTM accuracy
        'final_accuracy',           # After reversals
        'improvement',              # How much we improved
        'reversal_rate',           # Percentage of reversals
        'reversals_correct',       # Success rate of reversals
        'coverage'                 # Always 100% now!
    ]
    
    # ==================================================================================
    # EXPERIMENT TRACKING
    # ==================================================================================
    
    @property
    def OUTPUT_DIR(self):
        """Fixed output directory for meta model - flat structure"""
        return os.path.join(self.OUTPUT_BASE_DIR, f'meta_{self.FREQUENCY}_{self.MARKET}')
    
    @property
    def output_dir(self):
        """Alias for OUTPUT_DIR for compatibility"""
        return self.OUTPUT_DIR
    
    # ==================================================================================
    # VISUALIZATION
    # ==================================================================================
    
    # Plot settings
    PLOT_ERROR_DISTRIBUTION = True
    PLOT_REVERSAL_IMPACT = True
    PLOT_FEATURE_IMPORTANCE = True
    PLOT_THRESHOLD_OPTIMIZATION = True
    
    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    def __repr__(self):
        """String representation"""
        return (
            f"ReversalConfig(MARKET={self.MARKET}, "
            f"FREQUENCY={self.FREQUENCY}, "
            f"MODEL={self.MODEL_TYPE}, "
            f"COVERAGE={self.COVERAGE:.0%})"
        )
    
    def save(self, path):
        """Save configuration to JSON"""
        import json
        config_dict = {
            'market': self.MARKET,
            'frequency': self.FREQUENCY,
            'model_type': self.MODEL_TYPE,
            'coverage': self.COVERAGE,
            'reversal_threshold': self.INITIAL_REVERSAL_THRESHOLD,
            'max_reversal_rate': self.MAX_REVERSAL_RATE,
            'always_trade': self.ALWAYS_TRADE,
            'primary_model': self.PRIMARY_MODEL_DIR,
            'sequence_length': self.SEQUENCE_LENGTH
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)