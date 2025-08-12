"""
Configuration for Meta-Model that predicts LSTM prediction reliability
The meta-model uses sentiment features to predict if the primary LSTM's prediction will be correct
"""

import os
from datetime import datetime


class MetaConfig:
    """Configuration for meta-model training and evaluation"""
    
    def __init__(self, market='GDEA'):
        """Initialize config with specific market"""
        self.MARKET = market
    
    # ==================================================================================
    # EXPERIMENT CONFIGURATION
    # ==================================================================================
    
    # Primary LSTM model to evaluate
    @property
    def PRIMARY_MODEL_DIR(self):
        """Path to primary LSTM model (daily base model without sentiment)"""
        return f'../../04_Models/daily/{self.MARKET}/base'
    
    # ==================================================================================
    # DATA PATHS
    # ==================================================================================
    
    # Prepared LSTM sequences (to get exact split indices)
    LSTM_DATA_DIR = '../../02_Data_Processed/04_LSTM_Ready/'
    
    # Fixed output directory structure (no timestamps)
    OUTPUT_BASE_DIR = '../../04_Models/'
    
    @property
    def SENTIMENT_DATA_PATH(self):
        """Daily data with sentiment features"""
        return f'../../02_Data_Processed/10_Sentiment_Final_Merged/{self.MARKET}_LSTM_with_sentiment.parquet'
    
    @property
    def BASE_DATA_PATH(self):
        """Daily data without sentiment (for primary LSTM)"""
        return f'../../02_Data_Processed/03_Feature_Engineered/{self.MARKET}_LSTM_advanced.parquet'
    
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
    
    # Meta-model type (Random Forest with abstention is now the only option)
    META_MODEL_TYPE = 'random_forest'
    
    # Random Forest parameters - Conservative to prevent overfitting
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 30,           # Few trees to prevent memorization
        'max_depth': 3,               # Very shallow trees for generalization
        'min_samples_leaf': 10,       # Force general patterns
        'min_samples_split': 20,      # Require substantial splits
        'max_features': 'sqrt',       # Reduce correlation between trees
        'bootstrap': True,            # Enable out-of-bag scoring
        'criterion': 'gini',          # Split criterion
        'max_leaf_nodes': None,       # No limit on leaf nodes
        'min_weight_fraction_leaf': 0.0,
        'min_impurity_decrease': 0.0
    }
    
    # ==================================================================================
    # ABSTENTION CONFIGURATION
    # ==================================================================================
    
    # Abstention thresholds (relaxed to allow more trading after adding LSTM features)
    ABSTENTION_THRESHOLD = 0.30         # Max tree disagreement (reduced from 0.45)
    MIN_SENTIMENT_COVERAGE = 0.05       # Minimum fraction of features (reduced from 0.1)
    CONFIDENCE_MODE = 'tree_variance'   # How to calculate confidence
    
    # Confidence thresholds for trading (widened range)
    MIN_CONFIDENCE_TO_TRADE = 0.35      # Don't trade if confidence < 35% (was 40%)
    MAX_CONFIDENCE_TO_TRADE = 0.65      # Don't trade if confidence > 65% (was 60%)
    # (Trade in wider confidence range now that we have LSTM features)
    
    # ==================================================================================
    # ENSEMBLE STRATEGIES
    # ==================================================================================
    
    # Confidence threshold for filtered trading (now using abstention mechanism)
    CONFIDENCE_THRESHOLD = 0.55  # Trade when moderate confidence (not too high/low)
    
    # Position sizing based on confidence
    USE_POSITION_SIZING = True
    MIN_POSITION_SIZE = 0.2  # Minimum position when confidence is low
    MAX_POSITION_SIZE = 1.0  # Maximum position when confidence is high
    
    # Ensemble decision rules
    ENSEMBLE_STRATEGY = 'abstention_first'  # Default: prioritize abstention
    # Other options: 'filtered', 'selective', 'conservative', 'adaptive'
    
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
    def OUTPUT_DIR(self):
        """Fixed output directory for meta model"""
        return os.path.join(self.OUTPUT_BASE_DIR, 'meta', 'daily', self.MARKET)
    
    @property
    def output_dir(self):
        """Alias for OUTPUT_DIR for compatibility"""
        return self.OUTPUT_DIR
    
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
            'primary_model': self.PRIMARY_MODEL_DIR
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)