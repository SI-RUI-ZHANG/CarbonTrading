"""
Configuration file for Weekly LSTM prediction model
Supports both sentiment and non-sentiment versions through USE_SENTIMENT flag
"""

import torch
import os
from datetime import datetime

class Config:
    # ==================================================================================
    # EXPERIMENT CONFIGURATION
    # ==================================================================================
    
    # Market selection
    MARKET = 'HBEA'  # 'HBEA' or 'GDEA'
    
    # SENTIMENT FEATURE TOGGLE - KEY CONFIGURATION
    USE_SENTIMENT = True  # Set to False for baseline without sentiment
    
    def __init__(self):
        """Initialize config with fixed timestamp"""
        self._run_name = None
    
    # ==================================================================================
    # TIME AGGREGATION
    # ==================================================================================
    
    AGGREGATION = 'weekly'  # Time aggregation level
    SEQUENCE_LENGTH = 12    # 12 weeks lookback (~3 months of history)
    
    # ==================================================================================
    # DATA PATHS
    # ==================================================================================
    
    @property
    def input_daily_path(self):
        """Path to daily data with or without sentiment"""
        if self.USE_SENTIMENT:
            # Use merged data with sentiment features
            return f'../../02_Data_Processed/10_Sentiment_Final_Merged/{self.MARKET}_LSTM_with_sentiment.parquet'
        else:
            # Use original data without sentiment
            return f'../../02_Data_Processed/03_Feature_Engineered/{self.MARKET}_LSTM_advanced.parquet'
    
    @property
    def aggregated_data_path(self):
        """Path to save/load weekly aggregated data"""
        suffix = '_with_sentiment' if self.USE_SENTIMENT else ''
        return f'../../02_Data_Processed/11_Weekly_Aggregated/{self.MARKET}_weekly{suffix}.parquet'
    
    # Directory for processed LSTM-ready data
    DATA_DIR = '../../02_Data_Processed/12_LSTM_Weekly_Ready'
    
    # ==================================================================================
    # EXPERIMENT TRACKING
    # ==================================================================================
    
    @property
    def run_name(self):
        """Generate unique run name with configuration details (cached to avoid timestamp changes)"""
        if self._run_name is None:
            sentiment_tag = 'WithSentiment' if self.USE_SENTIMENT else 'NoSentiment'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._run_name = f"{timestamp}_{self.MARKET}_Weekly_{sentiment_tag}"
        return self._run_name
    
    BASE_OUTPUT_DIR = '../../04_Models_Weekly/'
    
    @property
    def output_dir(self):
        """Full output directory for this run"""
        return os.path.join(self.BASE_OUTPUT_DIR, self.run_name)
    
    # ==================================================================================
    # MODEL ARCHITECTURE
    # ==================================================================================
    
    # LSTM Architecture (shared between sentiment/no-sentiment)
    HIDDEN_SIZE = 64        # LSTM hidden units
    NUM_LAYERS = 2          # Number of LSTM layers
    DROPOUT = 0.2           # Dropout rate for regularization
    
    # Note: INPUT_SIZE will be determined dynamically based on features
    # With sentiment: ~63 features (50 base + 13 sentiment)
    # Without sentiment: 50 features
    
    # ==================================================================================
    # TRAINING CONFIGURATION
    # ==================================================================================
    
    # Adjusted for smaller weekly dataset
    BATCH_SIZE = 16         # Smaller batch size for ~380 training samples
    LEARNING_RATE = 0.001   # Standard learning rate
    NUM_EPOCHS = 150        # More epochs for smaller dataset
    EARLY_STOPPING_PATIENCE = 20  # More patience for weekly patterns
    
    # Training behavior
    SHUFFLE_TRAIN_LOADER = True  # Shuffle training data
    
    # ==================================================================================
    # DATA SPLITTING
    # ==================================================================================
    
    # Chronological split dates (same as daily but applied to weeks)
    TRAIN_END_DATE = '2020-12-31'
    VAL_END_DATE = '2022-12-31'
    
    # ==================================================================================
    # WEEKLY AGGREGATION PARAMETERS
    # ==================================================================================
    
    # Minimum trading days required for a valid week
    MIN_DAYS_PER_WEEK = 3
    
    # How to aggregate different feature types
    AGGREGATION_RULES = {
        # Price features
        'close': 'last',           # Friday close
        'vwap': 'mean',            # Average VWAP for week
        
        # Volume features  
        'volume_tons': 'sum',       # Total weekly volume
        'turnover_cny': 'sum',      # Total weekly turnover
        'cum_turnover_cny': 'last', # Cumulative at week end
        
        # Gap features
        'gap_days': 'min',         # Minimum gap in week
        
        # Macro features (already lagged, take last)
        'macro': 'last',            # Last value of week
        
        # Technical indicators - will be recalculated on weekly
        'technical': 'recalculate',
        
        # Sentiment features (if USE_SENTIMENT=True)
        'doc_count': 'sum',         # Total documents in week
        'sentiment_supply': 'weighted_mean',  # Weighted by doc_count
        'sentiment_demand': 'weighted_mean',  # Weighted by doc_count
        'sentiment_policy': 'weighted_mean',  # Weighted by doc_count
        'max_policy': 'max',       # Maximum policy strength
        'avg_policy': 'mean',      # Average policy strength
    }
    
    # ==================================================================================
    # DEVICE CONFIGURATION
    # ==================================================================================
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================================================================================
    # REPRODUCIBILITY
    # ==================================================================================
    
    SEED = 42
    
    # ==================================================================================
    # LOGGING
    # ==================================================================================
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    # ==================================================================================
    # FEATURE GROUPS (for aggregation logic)
    # ==================================================================================
    
    # Define feature groups for different aggregation rules
    PRICE_FEATURES = ['close', 'vwap']
    VOLUME_FEATURES = ['volume_tons', 'turnover_cny', 'cum_turnover_cny']
    GAP_FEATURES = ['gap_days', 'gap_days_age']
    
    # Macro features pattern (all contain 'ffill_daily')
    MACRO_PATTERN = 'ffill_daily'
    
    # Sentiment features (only if USE_SENTIMENT=True)
    SENTIMENT_FEATURES = [
        'sentiment_supply', 'sentiment_demand', 'sentiment_policy',
        'supply_decayed', 'demand_decayed', 'policy_decayed',
        'market_pressure', 'pressure_magnitude', 'news_shock',
        'pressure_momentum', 'supply_momentum', 'demand_momentum',
        'doc_count', 'max_policy', 'avg_policy'
    ]
    
    # Technical indicators to recalculate
    TECHNICAL_FEATURES = [
        'return_5d', 'bb_width', 'rsi_14', 'volume_sma_20'
    ]
    
    # Temporal features to recalculate
    TEMPORAL_FEATURES = [
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    
    # ==================================================================================
    # HELPER METHODS
    # ==================================================================================
    
    def get_feature_list(self):
        """Get list of features based on USE_SENTIMENT setting"""
        base_features = (
            self.PRICE_FEATURES + 
            self.VOLUME_FEATURES + 
            self.GAP_FEATURES +
            self.TECHNICAL_FEATURES +
            self.TEMPORAL_FEATURES
        )
        
        if self.USE_SENTIMENT:
            return base_features + self.SENTIMENT_FEATURES
        else:
            return base_features
    
    def __repr__(self):
        """String representation of configuration"""
        return (
            f"Config(MARKET={self.MARKET}, "
            f"USE_SENTIMENT={self.USE_SENTIMENT}, "
            f"AGGREGATION={self.AGGREGATION}, "
            f"SEQUENCE_LENGTH={self.SEQUENCE_LENGTH})"
        )

# Create global config instance
config = Config()