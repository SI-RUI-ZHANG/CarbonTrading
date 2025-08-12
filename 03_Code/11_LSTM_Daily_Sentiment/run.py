"""
Complete LSTM Training Pipeline for Daily Carbon Price Direction Prediction WITH SENTIMENT
Merges data preparation and model training into a single script
Uses sentiment features from policy documents
Supports both GDEA and HBEA markets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import logging
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import Config
from model_architecture import CarbonPriceLSTM
from utils import create_dataloaders, EarlyStopping, plot_training_history, save_config
from evaluate import evaluate_model, plot_predictions, calculate_metrics

# Import performance tracker
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from utils.performance_tracker import update_performance_summary
except ImportError:
    # Fallback if utils is not a package
    sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
    from performance_tracker import update_performance_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================================================
# ADDITIONAL CONFIGURATION (from data_preparation.py)
# ==================================================================================

# Constants for data preparation
SEQUENCE_LENGTH = 60  # 60-day lookback window
TRAIN_END_DATE = '2020-12-31'
VAL_END_DATE = '2022-12-31'
TARGET_COLUMN = 'log_return'  # Used to derive direction labels

# ==================================================================================
# DATA PREPARATION FUNCTIONS (from data_preparation.py)
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

def prepare_data(config, save_to_disk=False):
    """
    Complete data preparation pipeline
    
    Args:
        config: Configuration object
        save_to_disk: Whether to save prepared data to disk (for debugging)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("="*60)
    logger.info(f"DATA PREPARATION - {config.MARKET}")
    logger.info("="*60)
    
    # Construct input path (using sentiment features)
    input_path = f'../../02_Data_Processed/10_Sentiment_Final_Merged/{config.MARKET}_LSTM_with_sentiment.parquet'
    
    # Step 1: Load and clean data
    logger.info("\nStep 1: Loading and cleaning data...")
    df = load_and_clean_data(input_path)
    
    # Step 2: Chronological split
    logger.info("\nStep 2: Splitting data chronologically...")
    train_df, val_df, test_df = split_data_chronologically(
        df, TRAIN_END_DATE, VAL_END_DATE
    )
    
    # Step 3: Feature scaling
    logger.info("\nStep 3: Scaling features...")
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_df, val_df, test_df, TARGET_COLUMN
    )
    
    # Step 4: Sequence generation
    logger.info("\nStep 4: Creating sequences...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data(
        train_df, val_df, test_df,
        (train_scaled, val_scaled, test_scaled),
        TARGET_COLUMN, SEQUENCE_LENGTH
    )
    
    # Optional: Save to disk for debugging
    if save_to_disk:
        output_dir = config.DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\nSaving prepared data to {output_dir}...")
        
        # Save scaler
        joblib.dump(scaler, os.path.join(output_dir, f'{config.MARKET}_feature_scaler.pkl'))
        
        # Save arrays
        np.save(os.path.join(output_dir, f'{config.MARKET}_X_train.npy'), X_train)
        np.save(os.path.join(output_dir, f'{config.MARKET}_y_train.npy'), y_train)
        np.save(os.path.join(output_dir, f'{config.MARKET}_X_val.npy'), X_val)
        np.save(os.path.join(output_dir, f'{config.MARKET}_y_val.npy'), y_val)
        np.save(os.path.join(output_dir, f'{config.MARKET}_X_test.npy'), X_test)
        np.save(os.path.join(output_dir, f'{config.MARKET}_y_test.npy'), y_test)
        
        logger.info("Data saved to disk successfully")
    
    # Log summary statistics
    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"Sequence length: {SEQUENCE_LENGTH} days")
    logger.info(f"Number of features: {X_train.shape[2]}")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    logger.info(f"Test samples: {len(X_test):,}")
    logger.info(f"Total samples: {len(X_train) + len(X_val) + len(X_test):,}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================================================================================
# TRAINING FUNCTIONS (from model_training.py)
# ==================================================================================

def set_seed(seed):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, train_loader, criterion, optimizer, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(config.DEVICE)
        y_batch = y_batch.float().to(config.DEVICE)  # Float for BCEWithLogitsLoss
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Calculate accuracy using sigmoid threshold
        predicted = (torch.sigmoid(predictions) > 0.5).float()
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy

def validate_epoch(model, val_loader, criterion, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.float().to(config.DEVICE)  # Float for BCEWithLogitsLoss
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Calculate accuracy using sigmoid threshold
            predicted = (torch.sigmoid(predictions) > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

def train_model(config):
    """Main training function"""
    logger.info("\n" + "="*60)
    logger.info(f"MODEL TRAINING - {config.MARKET}")
    logger.info("="*60)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Device: {config.DEVICE}")
    
    # Set seed
    set_seed(config.SEED)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Prepare data (in-memory by default)
    logger.info("\nPreparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(config, save_to_disk=False)
    
    # DYNAMIC INPUT SIZE DETECTION
    config.INPUT_SIZE = X_train.shape[2]
    logger.info(f"\nDetected input size: {config.INPUT_SIZE} features")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Save configuration for reproducibility
    save_config(config)
    logger.info(f"Configuration saved to {config.OUTPUT_DIR}/config.json")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )
    logger.info(f"Train loader shuffle: {config.SHUFFLE_TRAIN_LOADER}")
    
    # Initialize model with dynamic input size for classification
    model = CarbonPriceLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate pos_weight for class imbalance
    num_negatives = np.sum(y_train == 0)
    num_positives = np.sum(y_train == 1)
    pos_weight_value = num_negatives / num_positives
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=config.DEVICE)
    logger.info(f"Class distribution - Down/Flat: {num_negatives}, Up: {num_positives}")
    logger.info(f"Calculated pos_weight for 'Up' class: {pos_weight_value:.2f}")
    
    # Loss and optimizer with class balancing
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_val_acc = 0
    
    logger.info("\nStarting training...")
    logger.info("-"*60)
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'input_size': config.INPUT_SIZE,
                'hidden_size': config.HIDDEN_SIZE,
                'num_layers': config.NUM_LAYERS,
                'dropout': config.DROPOUT
            }, f'{config.OUTPUT_DIR}/best_model.pth')
            logger.info(f"✓ Saved best model at epoch {epoch+1} (val_acc: {val_acc:.2f}%, val_loss: {val_loss:.6f})")
        
        # Logging
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                       f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info("-"*60)
    logger.info(f"Training completed. Total epochs: {len(train_losses)}")
    
    # Plot training history (updated for classification)
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, config)
    
    # Load best model for evaluation
    logger.info("Loading best model for evaluation...")
    checkpoint = torch.load(f'{config.OUTPUT_DIR}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictions, actuals, metrics, cm, probabilities = evaluate_model(model, test_loader, config)
    
    # Print metrics
    logger.info("="*60)
    logger.info("TEST SET CLASSIFICATION METRICS:")
    logger.info("-"*60)
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric:20s}: {value:.4f}")
        else:
            logger.info(f"  {metric:20s}: {value}")
    logger.info("="*60)
    
    # Save metrics
    with open(f'{config.OUTPUT_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot predictions with all classification visualization
    plot_predictions(predictions, actuals, metrics, cm, probabilities, config)
    
    # Save predictions
    np.save(f'{config.OUTPUT_DIR}/test_predictions.npy', predictions)
    np.save(f'{config.OUTPUT_DIR}/test_actuals.npy', actuals)
    
    logger.info(f"All results saved to: {config.OUTPUT_DIR}")
    
    # Update performance tracker
    update_performance_summary(
        model_type='daily_sentiment',
        market=config.MARKET,
        metrics=metrics,
        features_type='+Sentiment'
    )
    logger.info("Performance summary updated")
    
    return model, metrics

def main():
    """Main execution - runs both markets by default"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM Daily Model with Sentiment')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Store results for all markets
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("LSTM DAILY CARBON PRICE DIRECTION PREDICTION WITH SENTIMENT")
    logger.info(f"Markets to process: {', '.join(markets)}")
    logger.info("="*80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model for each market
    for market in markets:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING MARKET: {market}")
        logger.info("="*80)
        
        try:
            config = Config(market=market)
            model, metrics = train_model(config)
            all_metrics[market] = metrics
            logger.info(f"✅ {market} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {market} failed: {str(e)}")
            all_metrics[market] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    for market, metrics in all_metrics.items():
        if 'error' not in metrics:
            logger.info(f"{market}: Accuracy={metrics.get('test_accuracy', 0):.3f}, "
                       f"Precision={metrics.get('test_precision', 0):.3f}, "
                       f"Recall={metrics.get('test_recall', 0):.3f}, "
                       f"F1={metrics.get('test_f1', 0):.3f}")
        else:
            logger.info(f"{market}: Failed - {metrics['error']}")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ All markets completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    return all_metrics

if __name__ == "__main__":
    main()