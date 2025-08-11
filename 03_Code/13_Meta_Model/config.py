"""
Configuration for Meta-Model that predicts LSTM prediction reliability
The meta-model uses sentiment features to predict if the primary LSTM's prediction will be correct
"""

import os
from datetime import datetime


class MetaConfig:
    """Configuration for meta-model training and evaluation"""
    
    # ==================================================================================
    # EXPERIMENT CONFIGURATION
    # ==================================================================================
    
    # Market selection
    MARKET = 'HBEA'  # 'GDEA' or 'HBEA'
    
    # Primary LSTM model to evaluate
    PRIMARY_MODEL_DIR = '../../04_Models/20250810_155551_GDEA_LSTM_Classification'
    
    # If PRIMARY_MODEL_DIR doesn't exist, use these to find the best model
    PRIMARY_MODEL_BASE = '../../04_Models/'
    PRIMARY_MODEL_PATTERN = f'*{MARKET}_LSTM_Classification'
    
    # ==================================================================================
    # DATA PATHS
    # ==================================================================================
    
    # Daily data with sentiment features
    SENTIMENT_DATA_PATH = f'../../02_Data_Processed/10_Sentiment_Final_Merged/{MARKET}_LSTM_with_sentiment.parquet'
    
    # Daily data without sentiment (for primary LSTM)
    BASE_DATA_PATH = f'../../02_Data_Processed/03_Feature_Engineered/{MARKET}_LSTM_advanced.parquet'
    
    # Prepared LSTM sequences (to get exact split indices)
    LSTM_DATA_DIR = '../../02_Data_Processed/04_LSTM_Ready/'
    
    # Output directory for meta-model
    OUTPUT_BASE_DIR = '../../04_Models_Meta/'
    
    # ==================================================================================
    # SENTIMENT FEATURES FOR META-MODEL
    # ==================================================================================
    
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
    
    # Additional context features we might add
    CONTEXT_FEATURES = [
        # Could add: day_of_week, volatility, volume patterns, etc.
    ]
    
    # ==================================================================================
    # MODEL CONFIGURATION
    # ==================================================================================
    
    # Meta-model type
    META_MODEL_TYPE = 'xgboost'  # Options: 'xgboost', 'neural_net', 'logistic'
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'min_child_weight': 3,
        'seed': 42,
        'n_jobs': -1
    }
    
    # Neural network parameters (if used)
    NN_PARAMS = {
        'hidden_sizes': [32, 16],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 15
    }
    
    # ==================================================================================
    # ENSEMBLE STRATEGIES
    # ==================================================================================
    
    # Confidence threshold for filtered trading
    CONFIDENCE_THRESHOLD = 0.6  # Only trade when meta-model confidence > this
    
    # Position sizing based on confidence
    USE_POSITION_SIZING = True
    MIN_POSITION_SIZE = 0.2  # Minimum position when confidence is low
    MAX_POSITION_SIZE = 1.0  # Maximum position when confidence is high
    
    # Ensemble decision rules
    ENSEMBLE_STRATEGY = 'filtered'  # Options: 'filtered', 'weighted', 'selective', 'contrarian', 'volatility_gated'
    
    # ==================================================================================
    # TRAINING CONFIGURATION
    # ==================================================================================
    
    # Use same splits as primary LSTM
    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2
    
    # Sequence length (must match primary LSTM)
    SEQUENCE_LENGTH = 60
    
    # Random seed for reproducibility
    SEED = 42
    
    # ==================================================================================
    # EVALUATION METRICS
    # ==================================================================================
    
    # Metrics to track
    METRICS = [
        'accuracy',           # How well we predict if LSTM is correct
        'precision',          # When we say "trust", how often are we right
        'recall',            # How many correct predictions we identify
        'f1_score',
        'auc_roc',           # Overall discrimination ability
        'coverage',          # Percentage of trades we allow
        'filtered_accuracy', # Accuracy when we do trade
        'improvement'        # Improvement over baseline LSTM
    ]
    
    # ==================================================================================
    # EXPERIMENT TRACKING
    # ==================================================================================
    
    @property
    def run_name(self):
        """Generate unique run name"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{timestamp}_{self.MARKET}_Meta_{self.META_MODEL_TYPE}"
    
    @property
    def output_dir(self):
        """Full output directory for this run"""
        return os.path.join(self.OUTPUT_BASE_DIR, self.run_name)
    
    # ==================================================================================
    # VISUALIZATION
    # ==================================================================================
    
    # Plot settings
    PLOT_CONFIDENCE_DISTRIBUTION = True
    PLOT_FEATURE_IMPORTANCE = True
    PLOT_PERFORMANCE_COMPARISON = True
    PLOT_TRADING_SIGNALS = True
    
    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    def __repr__(self):
        """String representation"""
        return (
            f"MetaConfig(MARKET={self.MARKET}, "
            f"META_MODEL={self.META_MODEL_TYPE}, "
            f"STRATEGY={self.ENSEMBLE_STRATEGY})"
        )
    
    def save(self, path):
        """Save configuration to JSON"""
        import json
        config_dict = {
            'market': self.MARKET,
            'meta_model_type': self.META_MODEL_TYPE,
            'ensemble_strategy': self.ENSEMBLE_STRATEGY,
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'sentiment_features': self.SENTIMENT_FEATURES,
            'primary_model': self.PRIMARY_MODEL_DIR,
            'run_name': self.run_name
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Global instance
config = MetaConfig()