"""
Configuration for Weekly LSTM + Meta Model Ensemble
Supports both GDEA and HBEA markets with easy switching
"""

import os
import torch


class WeeklyConfig:
    """Configuration for weekly LSTM and meta model ensemble"""
    
    def __init__(self, market='GDEA'):
        """Initialize config with specific market"""
        self.MARKET = market
    
    # ==================================================================================
    # MARKET CONFIGURATION (Main switch between GDEA and HBEA)
    # ==================================================================================
    
    # ==================================================================================
    # EXPERIMENT CONFIGURATION
    # ==================================================================================
    
    # Use sentiment in base LSTM (for comparison, ensemble uses no-sentiment base)
    USE_SENTIMENT_IN_BASE = False  # Base LSTM for ensemble should NOT use sentiment
    
    # Fixed output directory structure (no timestamps)
    OUTPUT_BASE_DIR = '../../04_Models/'
    
    @property
    def output_dir(self):
        """Output directory for meta model"""
        return os.path.join(self.OUTPUT_BASE_DIR, 'meta', 'weekly', self.MARKET)
    
    # ==================================================================================
    # DATA PATHS
    # ==================================================================================
    
    @property
    def weekly_data_path(self):
        """Path to weekly aggregated data"""
        # For meta model, we always load the with_sentiment version
        # (base LSTM uses subset, meta model uses sentiment features)
        return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly_with_sentiment.parquet'
    
    @property
    def output_dir(self):
        """Output directory for this run"""
        base_dir = '../../04_Models_Meta_Weekly/'
        return os.path.join(base_dir, self.run_name)
    
    # ==================================================================================
    # WEEKLY LSTM CONFIGURATION
    # ==================================================================================
    
    # Sequence parameters
    SEQUENCE_LENGTH = 12  # 12 weeks lookback (~3 months)
    
    # Model architecture (same as 12_LSTM_Weekly)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    # Training parameters
    BATCH_SIZE = 16  # Small batch for ~380 training samples
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 20
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42
    
    # ==================================================================================
    # DATA SPLITTING (Chronological)
    # ==================================================================================
    
    TRAIN_END_DATE = '2020-12-31'
    VAL_END_DATE = '2022-12-31'
    
    # ==================================================================================
    # FEATURE CONFIGURATION
    # ==================================================================================
    
    # Base features (always included)
    BASE_FEATURES = [
        'close', 'vwap', 'volume_tons', 'turnover_cny', 'cum_turnover_cny',
        'gap_days', 'gap_days_age',
        # Technical indicators (recalculated weekly)
        'return_5w', 'bb_width', 'rsi_14', 'volume_sma_20',
        # Temporal
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    
    # Macro features (contain 'ffill_daily' in name)
    MACRO_PATTERN = 'ffill_daily'
    
    # Sentiment features (for meta model)
    SENTIMENT_FEATURES = [
        'doc_count',
        'sentiment_supply', 'sentiment_demand', 
        'supply_decayed', 'demand_decayed', 'policy_decayed',
        'market_pressure', 'pressure_magnitude', 'news_shock',
        'pressure_momentum', 'supply_momentum', 'demand_momentum',
        'max_policy', 'avg_policy'
    ]
    
    # ==================================================================================
    # RANDOM FOREST META MODEL CONFIGURATION
    # ==================================================================================
    
    # Random Forest parameters (tuned for weekly data)
    RF_PARAMS = {
        'n_estimators': 50,        # More trees for weekly patterns
        'max_depth': 4,            # Slightly deeper for complex patterns
        'min_samples_split': 5,    # Prevent overfitting on small data
        'min_samples_leaf': 3,     # Force generalization
        'max_features': 'sqrt',    # Reduce correlation between trees
        'random_state': SEED,
        'oob_score': True,         # Use OOB for honest assessment
        'n_jobs': -1               # Parallel processing
    }
    
    # ==================================================================================
    # ABSTENTION CONFIGURATION (Adjusted for weekly data)
    # ==================================================================================
    
    # Weekly data has better feature coverage due to aggregation
    MIN_SENTIMENT_COVERAGE = 0.05   # Lower threshold (was 0.1 for daily)
    ABSTENTION_THRESHOLD = 0.5      # Higher tolerance for disagreement (was 0.45)
    CONFIDENCE_MARGIN = 0.15        # Wider margin (was 0.1)
    
    # ==================================================================================
    # ENSEMBLE STRATEGY
    # ==================================================================================
    
    ENSEMBLE_STRATEGY = 'selective_abstention'  # Use meta model to selectively abstain
    
    # ==================================================================================
    # EVALUATION METRICS
    # ==================================================================================
    
    METRICS_TO_TRACK = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'coverage', 'abstention_rate', 
        'accuracy_when_trading',
        'improvement_over_base'
    ]
    
    # ==================================================================================
    # LOGGING
    # ==================================================================================
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    def get_base_features(self, df):
        """Extract base features from dataframe"""
        base_cols = []
        
        # Add explicit base features
        for col in self.BASE_FEATURES:
            if col in df.columns:
                base_cols.append(col)
        
        # Add macro features
        macro_cols = [c for c in df.columns if self.MACRO_PATTERN in c]
        base_cols.extend(macro_cols)
        
        return base_cols
    
    def get_sentiment_features(self, df):
        """Extract sentiment features from dataframe"""
        sent_cols = []
        for col in self.SENTIMENT_FEATURES:
            if col in df.columns:
                sent_cols.append(col)
        return sent_cols
    
    def get_all_features(self, df, use_sentiment=False):
        """Get all features based on configuration"""
        features = self.get_base_features(df)
        
        if use_sentiment:
            features.extend(self.get_sentiment_features(df))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for f in features:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)
        
        return unique_features
    
    def __repr__(self):
        return (
            f"WeeklyConfig(MARKET={self.MARKET}, "
            f"USE_SENTIMENT_IN_BASE={self.USE_SENTIMENT_IN_BASE}, "
            f"SEQUENCE_LENGTH={self.SEQUENCE_LENGTH})"
        )