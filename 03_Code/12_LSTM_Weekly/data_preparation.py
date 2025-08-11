"""
Weekly LSTM Data Preparation Pipeline
Transforms weekly aggregated data into sequences for LSTM training
Handles both sentiment and non-sentiment features based on configuration
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from typing import Tuple
import os
from datetime import datetime
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ==================================================================================
# DATA LOADING AND VALIDATION
# ==================================================================================

def load_weekly_data(file_path: str) -> pd.DataFrame:
    """
    Load weekly aggregated data
    
    Args:
        file_path: Path to the weekly parquet file
        
    Returns:
        DataFrame with weekly data
    """
    logger.info(f"Loading weekly data from {file_path}")
    df = pd.read_parquet(file_path)
    
    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape}")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Check for required columns
    required_cols = ['close', 'log_return', 'volume_tons']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Log feature groups
    logger.info("\nFeature Groups:")
    
    # Count different feature types
    price_cols = [c for c in df.columns if c in config.PRICE_FEATURES]
    volume_cols = [c for c in df.columns if c in config.VOLUME_FEATURES]
    macro_cols = [c for c in df.columns if config.MACRO_PATTERN in c]
    technical_cols = [c for c in df.columns if any(t in c for t in ['rsi', 'bb_width', 'return_5w', 'volume_sma'])]
    
    logger.info(f"  Price features: {len(price_cols)}")
    logger.info(f"  Volume features: {len(volume_cols)}")
    logger.info(f"  Macro features: {len(macro_cols)}")
    logger.info(f"  Technical features: {len(technical_cols)}")
    
    if config.USE_SENTIMENT:
        sentiment_cols = [c for c in df.columns if c in config.SENTIMENT_FEATURES]
        logger.info(f"  Sentiment features: {len(sentiment_cols)}")
    
    logger.info(f"  Total features: {len(df.columns)}")
    
    # Check for NaN values
    nan_counts = df.isna().sum().sum()
    if nan_counts > 0:
        logger.warning(f"Found {nan_counts} NaN values, will be dropped")
        df = df.dropna()
        logger.info(f"After cleaning: {df.shape}")
    
    return df

# ==================================================================================
# DATA SPLITTING
# ==================================================================================

def split_data_chronologically(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets
    
    Args:
        df: Input DataFrame with datetime index
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data chronologically")
    
    train_df = df[df.index <= config.TRAIN_END_DATE]
    val_df = df[(df.index > config.TRAIN_END_DATE) & (df.index <= config.VAL_END_DATE)]
    test_df = df[df.index > config.VAL_END_DATE]
    
    # Log split information
    logger.info(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} weeks)")
    logger.info(f"Val:   {val_df.index.min().date()} to {val_df.index.max().date()} ({len(val_df)} weeks)")
    logger.info(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} weeks)")
    
    # Log class distribution for each split
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if 'log_return' in split_df.columns:
            up_count = (split_df['log_return'] > 0).sum()
            total = len(split_df)
            logger.info(f"  {name} UP rate: {up_count}/{total} ({up_count/total*100:.1f}%)")
    
    # Verify no data leakage
    assert train_df.index.max() < val_df.index.min(), "Data leakage: Training data overlaps with validation"
    assert val_df.index.max() < test_df.index.min(), "Data leakage: Validation data overlaps with test"
    
    return train_df, val_df, test_df

# ==================================================================================
# FEATURE SCALING
# ==================================================================================

def scale_features(train_df: pd.DataFrame, 
                  val_df: pd.DataFrame, 
                  test_df: pd.DataFrame,
                  target_col: str = 'log_return') -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features using MinMaxScaler fitted on training data only
    
    Args:
        train_df, val_df, test_df: DataFrames for each split
        target_col: Name of target column to exclude from features
        
    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, fitted_scaler)
    """
    logger.info("Scaling features")
    
    # Identify feature columns (all except target)
    feature_cols = [col for col in train_df.columns if col != target_col]
    logger.info(f"Number of features to scale: {len(feature_cols)}")
    
    # Log which features are being scaled
    if config.USE_SENTIMENT:
        sentiment_features_present = [c for c in feature_cols if c in config.SENTIMENT_FEATURES]
        logger.info(f"  Including {len(sentiment_features_present)} sentiment features")
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler on training features only
    logger.info("Fitting scaler on training data only...")
    train_features_scaled = scaler.fit_transform(train_df[feature_cols])
    
    # Transform validation and test sets
    val_features_scaled = scaler.transform(val_df[feature_cols])
    test_features_scaled = scaler.transform(test_df[feature_cols])
    
    # Verify scaling
    logger.info(f"Scaled training range: [{train_features_scaled.min():.3f}, {train_features_scaled.max():.3f}]")
    logger.info(f"Scaled validation range: [{val_features_scaled.min():.3f}, {val_features_scaled.max():.3f}]")
    logger.info(f"Scaled test range: [{test_features_scaled.min():.3f}, {test_features_scaled.max():.3f}]")
    
    return train_features_scaled, val_features_scaled, test_features_scaled, scaler

# ==================================================================================
# SEQUENCE GENERATION
# ==================================================================================

def create_sequences(features: np.ndarray, 
                    targets: np.ndarray, 
                    sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using sliding window approach
    
    Args:
        features: 2D array of features (samples, features)
        targets: 1D array of target values
        sequence_length: Number of timesteps to look back
        
    Returns:
        X: 3D array (samples, sequence_length, features)
        y: 1D array (samples,)
    """
    X, y = [], []
    
    # Create sequences
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
    return X, y

def create_direction_labels(log_returns: np.ndarray) -> np.ndarray:
    """
    Convert log returns to binary direction labels
    
    Args:
        log_returns: Array of log returns
        
    Returns:
        Binary labels: 0 for down/flat (<=0), 1 for up (>0)
    """
    # Create binary labels
    direction_labels = (log_returns > 0).astype(np.int64)
    
    # Log class distribution
    n_down = np.sum(direction_labels == 0)
    n_up = np.sum(direction_labels == 1)
    total = len(direction_labels)
    
    logger.info(f"Direction labels distribution:")
    logger.info(f"  Down/Flat (0): {n_down} ({n_down/total*100:.1f}%)")
    logger.info(f"  Up (1): {n_up} ({n_up/total*100:.1f}%)")
    
    if n_down > 0:
        logger.info(f"  Class ratio (up/down): {n_up/n_down:.2f}")
    
    return direction_labels

def prepare_lstm_data(train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     features_scaled: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     target_col: str = 'log_return') -> Tuple[np.ndarray, ...]:
    """
    Create LSTM sequences for all datasets
    
    Args:
        train_df, val_df, test_df: Original DataFrames
        features_scaled: Tuple of scaled feature arrays
        target_col: Name of target column
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Creating sequences with lookback={config.SEQUENCE_LENGTH} weeks")
    
    train_scaled, val_scaled, test_scaled = features_scaled
    
    # Extract log returns and convert to direction labels
    logger.info("\nTraining set:")
    train_targets = create_direction_labels(train_df[target_col].values)
    
    logger.info("\nValidation set:")
    val_targets = create_direction_labels(val_df[target_col].values)
    
    logger.info("\nTest set:")
    test_targets = create_direction_labels(test_df[target_col].values)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, train_targets, config.SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_scaled, val_targets, config.SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_scaled, test_targets, config.SEQUENCE_LENGTH)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================================================================================
# SAVING ARTIFACTS
# ==================================================================================

def save_artifacts(scaler: MinMaxScaler,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
    """
    Save scaler and numpy arrays for model training
    """
    # Create output directory
    os.makedirs(config.DATA_DIR, exist_ok=True)
    logger.info(f"Saving artifacts to {config.DATA_DIR}")
    
    # Generate file prefix based on configuration
    sentiment_tag = 'sentiment' if config.USE_SENTIMENT else 'nosent'
    prefix = f'{config.MARKET}_weekly_{sentiment_tag}'
    
    # Save scaler
    scaler_path = os.path.join(config.DATA_DIR, f'{prefix}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save arrays
    arrays = {
        f'{prefix}_X_train.npy': X_train,
        f'{prefix}_y_train.npy': y_train,
        f'{prefix}_X_val.npy': X_val,
        f'{prefix}_y_val.npy': y_val,
        f'{prefix}_X_test.npy': X_test,
        f'{prefix}_y_test.npy': y_test
    }
    
    for filename, array in arrays.items():
        path = os.path.join(config.DATA_DIR, filename)
        np.save(path, array)
        logger.info(f"Saved {filename}: shape {array.shape}, dtype {array.dtype}")

# ==================================================================================
# VERIFICATION
# ==================================================================================

def verify_outputs() -> bool:
    """Verify all required files exist"""
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)
    
    sentiment_tag = 'sentiment' if config.USE_SENTIMENT else 'nosent'
    prefix = f'{config.MARKET}_weekly_{sentiment_tag}'
    
    required_files = [
        f'{prefix}_scaler.pkl',
        f'{prefix}_X_train.npy', f'{prefix}_y_train.npy',
        f'{prefix}_X_val.npy', f'{prefix}_y_val.npy',
        f'{prefix}_X_test.npy', f'{prefix}_y_test.npy'
    ]
    
    all_exist = True
    for filename in required_files:
        path = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(path):
            if filename.endswith('.npy'):
                array = np.load(path)
                logger.info(f"✓ {filename}: shape {array.shape}")
            else:
                logger.info(f"✓ {filename}: exists")
        else:
            logger.error(f"✗ {filename}: MISSING")
            all_exist = False
    
    return all_exist

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    """Main execution pipeline"""
    logger.info("="*80)
    logger.info(f"WEEKLY LSTM DATA PREPARATION - {config.MARKET}")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Market: {config.MARKET}")
    logger.info(f"  Use Sentiment: {config.USE_SENTIMENT}")
    logger.info(f"  Sequence Length: {config.SEQUENCE_LENGTH} weeks")
    logger.info(f"  Data Path: {config.aggregated_data_path}")
    
    try:
        # Step 1: Load weekly data
        logger.info("\n" + "-"*60)
        logger.info("STEP 1: LOADING WEEKLY DATA")
        logger.info("-"*60)
        df = load_weekly_data(config.aggregated_data_path)
        
        # Step 2: Chronological split
        logger.info("\n" + "-"*60)
        logger.info("STEP 2: CHRONOLOGICAL SPLITTING")
        logger.info("-"*60)
        train_df, val_df, test_df = split_data_chronologically(df)
        
        # Step 3: Feature scaling
        logger.info("\n" + "-"*60)
        logger.info("STEP 3: FEATURE SCALING")
        logger.info("-"*60)
        train_scaled, val_scaled, test_scaled, scaler = scale_features(
            train_df, val_df, test_df
        )
        
        # Step 4: Sequence generation
        logger.info("\n" + "-"*60)
        logger.info("STEP 4: SEQUENCE GENERATION")
        logger.info("-"*60)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data(
            train_df, val_df, test_df,
            (train_scaled, val_scaled, test_scaled)
        )
        
        # Step 5: Save artifacts
        logger.info("\n" + "-"*60)
        logger.info("STEP 5: SAVING ARTIFACTS")
        logger.info("-"*60)
        save_artifacts(
            scaler,
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 6: Verification
        success = verify_outputs()
        
        if success:
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("SUMMARY STATISTICS")
            logger.info("="*60)
            logger.info(f"Sequence length: {config.SEQUENCE_LENGTH} weeks")
            logger.info(f"Number of features: {X_train.shape[2]}")
            logger.info(f"Training samples: {len(X_train):,}")
            logger.info(f"Validation samples: {len(X_val):,}")
            logger.info(f"Test samples: {len(X_test):,}")
            
            # Data distribution
            total_samples = len(X_train) + len(X_val) + len(X_test)
            logger.info(f"Total samples: {total_samples:,}")
            logger.info(f"  Train: {len(X_train)/total_samples*100:.1f}%")
            logger.info(f"  Val:   {len(X_val)/total_samples*100:.1f}%")
            logger.info(f"  Test:  {len(X_test)/total_samples*100:.1f}%")
            
            logger.info("\n✅ Pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()