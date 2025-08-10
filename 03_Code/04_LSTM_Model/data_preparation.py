"""
LSTM Data Preparation Pipeline
Transforms 2D feature DataFrame into 3D sequences for LSTM training

This script:
1. Loads HBEA_LSTM_advanced.parquet with 52 features
2. Removes rows with NaN values (from technical indicators)
3. Splits data chronologically (train: ~2020, val: 2021-2022, test: 2023+)
4. Scales features using MinMaxScaler fitted on training data only
5. Creates sliding window sequences (60-day lookback)
6. Saves processed arrays and scaler for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from typing import Tuple, List
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Constants
SEQUENCE_LENGTH = 60  # 60-day lookback window
TRAIN_END_DATE = '2020-12-31'
VAL_END_DATE = '2022-12-31'
TARGET_COLUMN = 'log_return'  # Used to derive direction labels
TASK_TYPE = 'classification'  # Changed from regression to classification
RANDOM_SEED = 42

MARKET = 'HBEA'  # HBEA or GDEA

# Paths
INPUT_PATH = f'../../02_Data_Processed/03_Feature_Engineered/{MARKET}_LSTM_advanced.parquet'
OUTPUT_DIR = f'../../02_Data_Processed/04_LSTM_Ready/'

# ==================================================================================
# DATA LOADING AND CLEANING
# ==================================================================================

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load data and remove NaN values
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Cleaned DataFrame with date index
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    
    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape}")
    
    # Ensure we have a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex, attempting to convert")
        df.index = pd.to_datetime(df.index)
    
    # Check NaN values
    nan_counts = df.isna().sum().sum()
    nan_rows = df.isna().any(axis=1).sum()
    logger.info(f"Found {nan_counts} NaN values across {nan_rows} rows")
    
    # Show which columns have NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.info(f"Columns with NaNs: {nan_cols[:5]}{'...' if len(nan_cols) > 5 else ''}")
    
    # Drop NaN rows (these are from the initial period of technical indicators)
    df_clean = df.dropna()
    final_shape = df_clean.shape
    logger.info(f"After cleaning: {final_shape} (removed {initial_shape[0] - final_shape[0]} rows)")
    
    return df_clean

# ==================================================================================
# DATA SPLITTING
# ==================================================================================

def split_data_chronologically(df: pd.DataFrame, 
                              train_end: str, 
                              val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets
    
    Args:
        df: Input DataFrame with datetime index
        train_end: Last date for training data
        val_end: Last date for validation data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data chronologically")
    
    train_df = df[df.index <= train_end]
    val_df = df[(df.index > train_end) & (df.index <= val_end)]
    test_df = df[df.index > val_end]
    
    # Log split information
    logger.info(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} days)")
    logger.info(f"Val:   {val_df.index.min().date()} to {val_df.index.max().date()} ({len(val_df)} days)")
    logger.info(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} days)")
    
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
                  target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features using MinMaxScaler fitted on training data only
    
    CRITICAL: Fit scaler on training data only to prevent data leakage
    
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
    
    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # CRITICAL: Fit scaler on training features only
    logger.info("Fitting scaler on training data only...")
    train_features_scaled = scaler.fit_transform(train_df[feature_cols])
    
    # Transform validation and test sets using the fitted scaler
    logger.info("Transforming validation and test data with fitted scaler...")
    val_features_scaled = scaler.transform(val_df[feature_cols])
    test_features_scaled = scaler.transform(test_df[feature_cols])
    
    # Verify scaling
    logger.info(f"Scaled training features range: [{train_features_scaled.min():.3f}, {train_features_scaled.max():.3f}]")
    logger.info(f"Scaled validation features range: [{val_features_scaled.min():.3f}, {val_features_scaled.max():.3f}]")
    logger.info(f"Scaled test features range: [{test_features_scaled.min():.3f}, {test_features_scaled.max():.3f}]")
    
    logger.info("Scaling completed successfully")
    return train_features_scaled, val_features_scaled, test_features_scaled, scaler

# ==================================================================================
# SEQUENCE GENERATION
# ==================================================================================

def create_sequences(features: np.ndarray, 
                    targets: np.ndarray, 
                    sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences using sliding window approach
    
    For each position i, create:
    - X: features[i-sequence_length:i] (lookback window)
    - y: targets[i] (next value to predict)
    
    Args:
        features: 2D array of features (samples, features)
        targets: 1D array of target values
        sequence_length: Number of timesteps to look back
        
    Returns:
        X: 3D array (samples, sequence_length, features)
        y: 1D array (samples,)
    """
    X, y = [], []
    
    # Start from sequence_length to ensure we have enough history
    for i in range(sequence_length, len(features)):
        # Look back sequence_length timesteps
        X.append(features[i-sequence_length:i])
        # Target is the value at position i
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
    # Create binary labels: 1 for positive returns, 0 for negative/zero
    direction_labels = (log_returns > 0).astype(np.int64)
    
    # Log class distribution
    n_down = np.sum(direction_labels == 0)
    n_up = np.sum(direction_labels == 1)
    total = len(direction_labels)
    
    logger.info(f"Direction labels distribution:")
    logger.info(f"  Down/Flat (0): {n_down} ({n_down/total*100:.1f}%)")
    logger.info(f"  Up (1): {n_up} ({n_up/total*100:.1f}%)")
    logger.info(f"  Class ratio (up/down): {n_up/n_down:.2f}")
    
    return direction_labels

def prepare_lstm_data(train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     features_scaled: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     target_col: str,
                     sequence_length: int) -> Tuple[np.ndarray, ...]:
    """
    Create LSTM sequences for all datasets with direction classification
    
    Args:
        train_df, val_df, test_df: Original DataFrames (for extracting targets)
        features_scaled: Tuple of scaled feature arrays
        target_col: Name of target column (log_return)
        sequence_length: Lookback window size
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        where y arrays contain direction labels (0 or 1)
    """
    logger.info(f"Creating sequences with lookback={sequence_length}")
    
    train_scaled, val_scaled, test_scaled = features_scaled
    
    # Extract log returns
    train_log_returns = train_df[target_col].values
    val_log_returns = val_df[target_col].values
    test_log_returns = test_df[target_col].values
    
    # Convert to direction labels
    logger.info("Converting log returns to direction labels...")
    logger.info("\nTraining set:")
    train_targets = create_direction_labels(train_log_returns)
    logger.info("\nValidation set:")
    val_targets = create_direction_labels(val_log_returns)
    logger.info("\nTest set:")
    test_targets = create_direction_labels(test_log_returns)
    
    # Create sequences for each split
    X_train, y_train = create_sequences(train_scaled, train_targets, sequence_length)
    X_val, y_val = create_sequences(val_scaled, val_targets, sequence_length)
    X_test, y_test = create_sequences(test_scaled, test_targets, sequence_length)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================================================================================
# SAVING ARTIFACTS
# ==================================================================================

def save_artifacts(output_dir: str,
                  market: str,
                  scaler: MinMaxScaler,
                  X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
    """
    Save scaler and numpy arrays for model training with market prefix
    
    Args:
        output_dir: Directory to save files
        market: Market identifier (HBEA or GDEA)
        scaler: Fitted MinMaxScaler
        X_train, y_train, X_val, y_val, X_test, y_test: Arrays to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving artifacts to {output_dir}")
    
    # Save scaler with market prefix
    scaler_path = os.path.join(output_dir, f'{market}_feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Save arrays with market prefix
    arrays = {
        f'{market}_X_train.npy': X_train,
        f'{market}_y_train.npy': y_train,
        f'{market}_X_val.npy': X_val,
        f'{market}_y_val.npy': y_val,
        f'{market}_X_test.npy': X_test,
        f'{market}_y_test.npy': y_test
    }
    
    for filename, array in arrays.items():
        path = os.path.join(output_dir, filename)
        np.save(path, array)
        logger.info(f"Saved {filename}: shape {array.shape}, dtype {array.dtype}")

# ==================================================================================
# VERIFICATION
# ==================================================================================

def verify_outputs(output_dir: str, market: str) -> bool:
    """
    Verify all required files exist and log their properties
    
    Args:
        output_dir: Directory containing saved files
        market: Market identifier (HBEA or GDEA)
        
    Returns:
        True if all files exist, False otherwise
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION")
    logger.info("="*60)
    
    required_files = [
        f'{market}_feature_scaler.pkl',
        f'{market}_X_train.npy', f'{market}_y_train.npy',
        f'{market}_X_val.npy', f'{market}_y_val.npy',
        f'{market}_X_test.npy', f'{market}_y_test.npy'
    ]
    
    all_exist = True
    for filename in required_files:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            if filename.endswith('.npy'):
                array = np.load(path)
                logger.info(f"✓ {filename}: shape {array.shape}, dtype {array.dtype}")
                
                # Additional checks for array validity
                if 'X_' in filename:
                    if len(array.shape) != 3:
                        logger.warning(f"  WARNING: Expected 3D array for {filename}, got {len(array.shape)}D")
                elif 'y_' in filename:
                    if len(array.shape) != 1:
                        logger.warning(f"  WARNING: Expected 1D array for {filename}, got {len(array.shape)}D")
                    # Check for classification labels
                    unique_vals = np.unique(array)
                    if len(unique_vals) <= 3:
                        logger.info(f"    Classification labels: {unique_vals}")
            else:
                # Verify scaler can be loaded
                try:
                    loaded_scaler = joblib.load(path)
                    logger.info(f"✓ {filename}: MinMaxScaler with {loaded_scaler.n_features_in_} features")
                except Exception as e:
                    logger.error(f"  ERROR loading scaler: {e}")
                    all_exist = False
        else:
            logger.error(f"✗ {filename}: MISSING")
            all_exist = False
    
    if all_exist:
        logger.info(f"\n✅ All required files for {market} successfully created and verified!")
    else:
        logger.error(f"\n❌ Some files for {market} are missing or invalid!")
    
    return all_exist

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    """
    Main execution pipeline
    """
    logger.info("="*60)
    logger.info(f"LSTM DATA PREPARATION PIPELINE - {MARKET}")
    logger.info("="*60)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  - Market: {MARKET}")
    logger.info(f"  - Task type: {TASK_TYPE}")
    logger.info(f"  - Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"  - Target column: {TARGET_COLUMN} (converted to direction)")
    logger.info(f"  - Train end date: {TRAIN_END_DATE}")
    logger.info(f"  - Val end date: {VAL_END_DATE}")
    
    try:
        # Step 1: Load and clean data
        logger.info("\n" + "-"*60)
        logger.info("STEP 1: DATA LOADING & CLEANING")
        logger.info("-"*60)
        df = load_and_clean_data(INPUT_PATH)
        
        # Step 2: Chronological split
        logger.info("\n" + "-"*60)
        logger.info("STEP 2: CHRONOLOGICAL SPLITTING")
        logger.info("-"*60)
        train_df, val_df, test_df = split_data_chronologically(
            df, TRAIN_END_DATE, VAL_END_DATE
        )
        
        # Step 3: Feature scaling
        logger.info("\n" + "-"*60)
        logger.info("STEP 3: FEATURE SCALING")
        logger.info("-"*60)
        train_scaled, val_scaled, test_scaled, scaler = scale_features(
            train_df, val_df, test_df, TARGET_COLUMN
        )
        
        # Step 4: Sequence generation
        logger.info("\n" + "-"*60)
        logger.info("STEP 4: SEQUENCE GENERATION")
        logger.info("-"*60)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data(
            train_df, val_df, test_df,
            (train_scaled, val_scaled, test_scaled),
            TARGET_COLUMN, SEQUENCE_LENGTH
        )
        
        # Step 5: Save artifacts
        logger.info("\n" + "-"*60)
        logger.info("STEP 5: SAVING ARTIFACTS")
        logger.info("-"*60)
        save_artifacts(
            OUTPUT_DIR, MARKET, scaler,
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 6: Verification
        logger.info("\n" + "-"*60)
        logger.info("STEP 6: FINAL VERIFICATION")
        logger.info("-"*60)
        success = verify_outputs(OUTPUT_DIR, MARKET)
        
        if success:
            # Print summary statistics
            logger.info("\n" + "="*60)
            logger.info("SUMMARY STATISTICS")
            logger.info("="*60)
            logger.info(f"Sequence length: {SEQUENCE_LENGTH} days")
            logger.info(f"Number of features: {X_train.shape[2]}")
            logger.info(f"Training samples: {len(X_train):,}")
            logger.info(f"Validation samples: {len(X_val):,}")
            logger.info(f"Test samples: {len(X_test):,}")
            logger.info(f"Total samples: {len(X_train) + len(X_val) + len(X_test):,}")
            
            # Data distribution
            total_samples = len(X_train) + len(X_val) + len(X_test)
            train_pct = len(X_train) / total_samples * 100
            val_pct = len(X_val) / total_samples * 100
            test_pct = len(X_test) / total_samples * 100
            logger.info(f"\nData distribution:")
            logger.info(f"  Train: {train_pct:.1f}%")
            logger.info(f"  Val:   {val_pct:.1f}%")
            logger.info(f"  Test:  {test_pct:.1f}%")
            
            # Memory usage
            total_memory = (X_train.nbytes + y_train.nbytes + 
                          X_val.nbytes + y_val.nbytes + 
                          X_test.nbytes + y_test.nbytes) / (1024**2)
            logger.info(f"\nTotal memory usage: {total_memory:.2f} MB")
            
            logger.info("\n" + "="*60)
            logger.info(f"✅ Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*60)
            
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()