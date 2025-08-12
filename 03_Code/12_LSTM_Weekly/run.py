"""
Complete Weekly LSTM Training Pipeline
Merges data preparation and model training into a single script
Handles both sentiment and non-sentiment versions based on config
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List
import joblib
from sklearn.preprocessing import MinMaxScaler

from config import Config
from model import create_model, EarlyStopping
from visualization import plot_training_history, create_classification_analysis, save_config

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================================================
# DATA PREPARATION FUNCTIONS (from data_preparation.py)
# ==================================================================================

def load_weekly_data(config) -> pd.DataFrame:
    """
    Load weekly aggregated data based on configuration
    
    Returns:
        DataFrame with weekly data
    """
    # Determine which file to load based on sentiment flag
    if config.USE_SENTIMENT:
        data_path = os.path.join(config.AGGREGATED_DATA_DIR, f'{config.MARKET}_weekly_with_sentiment.parquet')
    else:
        data_path = os.path.join(config.AGGREGATED_DATA_DIR, f'{config.MARKET}_weekly.parquet')
    
    logger.info(f"Loading weekly data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    logger.info(f"Loaded {len(df)} weeks of data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Total features: {len(df.columns)}")
    
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target for price direction prediction
    
    Args:
        df: DataFrame with weekly data
        
    Returns:
        DataFrame with added target column
    """
    # Target is next week's direction
    df['target'] = (df['log_return'].shift(-1) > 0).astype(int)
    
    # Drop last row (no target)
    df = df[:-1]
    
    # Log class distribution
    up_count = (df['target'] == 1).sum()
    down_count = (df['target'] == 0).sum()
    logger.info(f"Target distribution - Up: {up_count} ({up_count/len(df):.1%}), "
               f"Down/Flat: {down_count} ({down_count/len(df):.1%})")
    
    return df

def select_features(df: pd.DataFrame, config) -> List[str]:
    """
    Select features based on configuration
    
    Args:
        df: DataFrame with all features
        use_sentiment: Whether to include sentiment features
        
    Returns:
        List of feature column names
    """
    # Get features based on sentiment flag
    feature_columns = list(df.columns)
    
    # Remove target and return columns
    exclude = ['target', 'log_return', 'return']
    feature_columns = [f for f in feature_columns if f not in exclude]
    
    logger.info(f"Selected {len(feature_columns)} features")
    
    if config.USE_SENTIMENT:
        # Count sentiment features (columns with 'sentiment' in name)
        sentiment_cols = [col for col in feature_columns if 'sentiment' in col.lower() or 'policy' in col.lower() or 'supply' in col.lower() or 'demand' in col.lower()]
        logger.info(f"  Including {len(sentiment_cols)} sentiment features")
    
    return feature_columns

def split_data(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test
    
    Args:
        df: DataFrame with all data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df[df.index <= config.TRAIN_END_DATE]
    val_df = df[(df.index > config.TRAIN_END_DATE) & (df.index <= config.VAL_END_DATE)]
    test_df = df[df.index > config.VAL_END_DATE]
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"  Train: {train_df.index[0]} to {train_df.index[-1]}")
    logger.info(f"  Val: {val_df.index[0]} to {val_df.index[-1]}")
    logger.info(f"  Test: {test_df.index[0]} to {test_df.index[-1]}")
    
    return train_df, val_df, test_df

def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale features using MinMaxScaler fitted on training data
    
    Args:
        train_df, val_df, test_df: Split dataframes
        feature_columns: List of feature column names
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on training data
    X_train = train_df[feature_columns].values
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation and test
    X_val = val_df[feature_columns].values
    X_val_scaled = scaler.transform(X_val)
    
    X_test = test_df[feature_columns].values
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled using MinMaxScaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def create_sequences(X: np.ndarray, y: np.ndarray, 
                    sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input
    
    Args:
        X: Feature array
        y: Target array
        sequence_length: Number of timesteps to look back
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)

def prepare_lstm_data(config, save_to_disk: bool = False) -> Tuple[np.ndarray, ...]:
    """
    Complete pipeline to prepare LSTM data
    
    Args:
        save_to_disk: Whether to save prepared data to disk (for debugging)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("="*60)
    logger.info(f"DATA PREPARATION - {config.MARKET} Weekly")
    logger.info("="*60)
    
    # Load weekly data
    df = load_weekly_data(config)
    
    # Create target
    df = create_target(df)
    
    # Select features
    feature_columns = select_features(df, config)
    
    # Split data
    train_df, val_df, test_df = split_data(df, config)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        train_df, val_df, test_df, feature_columns
    )
    
    # Get targets
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    y_test = test_df['target'].values
    
    # Create sequences
    logger.info(f"Creating sequences with lookback={config.SEQUENCE_LENGTH}")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, config.SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, config.SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, config.SEQUENCE_LENGTH)
    
    logger.info(f"Sequence shapes:")
    logger.info(f"  Train: X={X_train_seq.shape}, y={y_train_seq.shape}")
    logger.info(f"  Val: X={X_val_seq.shape}, y={y_val_seq.shape}")
    logger.info(f"  Test: X={X_test_seq.shape}, y={y_test_seq.shape}")
    
    # Optional: Save to disk for debugging
    if save_to_disk:
        output_dir = config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        sentiment_tag = 'sentiment' if config.USE_SENTIMENT else 'nosent'
        prefix = f'{config.MARKET}_weekly_{sentiment_tag}'
        
        # Save arrays
        np.save(os.path.join(output_dir, f'{prefix}_X_train.npy'), X_train_seq)
        np.save(os.path.join(output_dir, f'{prefix}_y_train.npy'), y_train_seq)
        np.save(os.path.join(output_dir, f'{prefix}_X_val.npy'), X_val_seq)
        np.save(os.path.join(output_dir, f'{prefix}_y_val.npy'), y_val_seq)
        np.save(os.path.join(output_dir, f'{prefix}_X_test.npy'), X_test_seq)
        np.save(os.path.join(output_dir, f'{prefix}_y_test.npy'), y_test_seq)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(output_dir, f'{prefix}_scaler.pkl'))
        
        logger.info(f"Data saved to {output_dir}")
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq

# ==================================================================================
# DATA PREPARATION FOR TRAINING
# ==================================================================================

def prepare_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        config) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # Add dimension for BCELoss
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN_LOADER
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, val_loader

# ==================================================================================
# TRAINING FUNCTIONS (from train.py)
# ==================================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, config) -> Tuple[float, float]:
    """
    Train for one epoch
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(config.DEVICE)
        y_batch = y_batch.to(config.DEVICE)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate_epoch(model: nn.Module, val_loader: DataLoader, 
                  criterion: nn.Module, config) -> Tuple[float, float]:
    """
    Validate for one epoch
    
    Args:
        model: LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, config) -> Dict:
    """
    Main training function
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Prepared data
        
    Returns:
        Dictionary with metrics and results
    """
    logger.info("\n" + "="*60)
    logger.info("MODEL TRAINING")
    logger.info("="*60)
    
    # Create dataloaders
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, config
    )
    
    # Create model
    input_size = X_train.shape[2]
    model = create_model(input_size, config)
    
    # Calculate class weights
    num_negatives = np.sum(y_train == 0)
    num_positives = np.sum(y_train == 1)
    pos_weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], device=config.DEVICE)
    
    logger.info(f"Class distribution - Down/Flat: {num_negatives}, Up: {num_positives}")
    logger.info(f"Calculated pos_weight: {pos_weight_value:.2f}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0
    best_epoch = 0
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')  # Initialize path
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model (best_model_path already initialized)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config.__dict__
            }, best_model_path)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")
        
        # Early stopping
        early_stopping(val_loss, model, best_model_path)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.3f} at epoch {best_epoch+1}")
    
    # Load best model
    best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        # Handle both checkpoint dict and direct state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's a direct state dict (from EarlyStopping)
            model.load_state_dict(checkpoint)
    else:
        logger.warning("Best model checkpoint not found, using current model state")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    model.eval()
    
    # Prepare test data
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor.unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Test evaluation
    test_predictions = []
    test_actuals = []
    test_probabilities = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            test_predictions.extend(preds.cpu().numpy().flatten())
            test_actuals.extend(y_batch.numpy().flatten())
            test_probabilities.extend(probs.cpu().numpy().flatten())
    
    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)
    test_probabilities = np.array(test_probabilities)
    
    # Calculate metrics
    test_accuracy = np.mean(test_predictions == test_actuals)
    
    # Calculate precision, recall, F1
    tp = np.sum((test_predictions == 1) & (test_actuals == 1))
    fp = np.sum((test_predictions == 1) & (test_actuals == 0))
    fn = np.sum((test_predictions == 0) & (test_actuals == 1))
    tn = np.sum((test_predictions == 0) & (test_actuals == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'test_accuracy': test_accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch + 1,
        'total_epochs': len(train_losses)
    }
    
    logger.info(f"Test Accuracy: {test_accuracy:.3f}")
    logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Save results
    # Save metrics
    with open(os.path.join(config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and probabilities
    np.save(os.path.join(config.OUTPUT_DIR, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(config.OUTPUT_DIR, 'test_actuals.npy'), test_actuals)
    np.save(os.path.join(config.OUTPUT_DIR, 'test_probabilities.npy'), test_probabilities)
    
    # Save configuration
    save_config(config, os.path.join(config.OUTPUT_DIR, 'config.json'))
    
    # Plot training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    plot_training_history(history, config, save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png'))
    
    # Create classification analysis
    create_classification_analysis(
        y_true=test_actuals,
        y_pred=test_predictions, 
        y_prob=test_probabilities,
        metrics=metrics,
        config=config,
        save_path=os.path.join(config.OUTPUT_DIR, 'classification_analysis.png')
    )
    
    logger.info(f"All results saved to: {config.OUTPUT_DIR}")
    
    # Update performance tracker
    model_type = 'weekly_sentiment' if config.USE_SENTIMENT else 'weekly_base'
    features_type = '+Sentiment' if config.USE_SENTIMENT else 'Base'
    update_performance_summary(
        model_type=model_type,
        market=config.MARKET,
        metrics=metrics,
        features_type=features_type
    )
    logger.info("Performance summary updated")
    
    return metrics

def main():
    """Main execution - runs all combinations of markets and sentiment"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Weekly LSTM Model')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--sentiment', choices=['base', 'sentiment', 'both'],
                       default='both', help='Feature set(s) to use (default: both)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Determine which sentiment options to run
    if args.sentiment == 'both':
        sentiment_options = [False, True]  # Base first, then with sentiment
    elif args.sentiment == 'base':
        sentiment_options = [False]
    else:  # sentiment
        sentiment_options = [True]
    
    # Store results for all combinations
    all_metrics = {}
    
    logger.info("\n" + "="*80)
    logger.info("WEEKLY LSTM CARBON PRICE DIRECTION PREDICTION")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Features: {args.sentiment}")
    logger.info("="*80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train model for each combination
    for market in markets:
        for use_sentiment in sentiment_options:
            config_name = f"{market}_{'sentiment' if use_sentiment else 'base'}"
            
            logger.info("\n" + "="*80)
            logger.info(f"PROCESSING: {config_name}")
            logger.info("="*80)
            
            try:
                # Create config for this combination
                config = Config(market=market, use_sentiment=use_sentiment)
                logger.info(f"Market: {config.MARKET}")
                logger.info(f"Use Sentiment: {config.USE_SENTIMENT}")
                logger.info(f"Output: {config.output_dir}")
                
                # Prepare data (in-memory by default)
                X_train, y_train, X_val, y_val, X_test, y_test = prepare_lstm_data(config, save_to_disk=False)
                
                # Train model
                metrics = train_model(X_train, y_train, X_val, y_val, X_test, y_test, config)
                all_metrics[config_name] = metrics
                logger.info(f"✅ {config_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {config_name} failed: {str(e)}")
                all_metrics[config_name] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    for config_name, metrics in all_metrics.items():
        if 'error' not in metrics:
            logger.info(f"{config_name}: Accuracy={metrics.get('test_accuracy', 0):.3f}, "
                       f"Precision={metrics.get('test_precision', 0):.3f}, "
                       f"Recall={metrics.get('test_recall', 0):.3f}, "
                       f"F1={metrics.get('test_f1', 0):.3f}")
        else:
            logger.info(f"{config_name}: Failed - {metrics['error'][:50]}...")
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ All combinations completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    return all_metrics

if __name__ == "__main__":
    main()