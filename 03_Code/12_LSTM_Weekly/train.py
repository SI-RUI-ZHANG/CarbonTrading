"""
Training script for Weekly LSTM model
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
from typing import Tuple, Dict
import joblib

from config import config
from model import create_model, EarlyStopping
from visualization import plot_training_history, create_classification_analysis, save_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ==================================================================================
# DATA LOADING
# ==================================================================================

def load_data() -> Tuple[np.ndarray, ...]:
    """
    Load prepared LSTM data based on configuration
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Match the naming convention from data_preparation.py
    sentiment_tag = 'sentiment' if config.USE_SENTIMENT else 'nosent'
    prefix = f'{config.MARKET}_weekly_{sentiment_tag}'
    
    logger.info(f"Loading data for {config.MARKET} with sentiment={config.USE_SENTIMENT}")
    
    # Load numpy arrays
    X_train = np.load(os.path.join(config.DATA_DIR, f'{prefix}_X_train.npy'))
    y_train = np.load(os.path.join(config.DATA_DIR, f'{prefix}_y_train.npy'))
    X_val = np.load(os.path.join(config.DATA_DIR, f'{prefix}_X_val.npy'))
    y_val = np.load(os.path.join(config.DATA_DIR, f'{prefix}_y_val.npy'))
    X_test = np.load(os.path.join(config.DATA_DIR, f'{prefix}_X_test.npy'))
    y_test = np.load(os.path.join(config.DATA_DIR, f'{prefix}_y_test.npy'))
    
    logger.info(f"Data shapes:")
    logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================================================================================
# DATA PREPARATION
# ==================================================================================

def prepare_dataloaders(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
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
        batch_size=batch_size,
        shuffle=config.SHUFFLE_TRAIN_LOADER
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

# ==================================================================================
# TRAINING LOOP
# ==================================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    Train for one epoch
    
    Args:
        model: LSTM model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(config.DEVICE)
        target = target.to(config.DEVICE)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """
    Validate the model
    
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
        for data, target in val_loader:
            data = data.to(config.DEVICE)
            target = target.to(config.DEVICE)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (torch.sigmoid(output) >= 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# ==================================================================================
# MAIN TRAINING
# ==================================================================================

def train_model():
    """Main training function"""
    
    logger.info("="*80)
    logger.info(f"WEEKLY LSTM TRAINING - {config.MARKET}")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Market: {config.MARKET}")
    logger.info(f"  Use Sentiment: {config.USE_SENTIMENT}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    logger.info(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    
    # Create output directory - ensure 04_Models exists first
    os.makedirs(config.BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Output directory: {config.output_dir}")
    
    # Save configuration
    save_config(config, os.path.join(config.output_dir, 'config.json'))
    logger.info("Configuration saved")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Load data
    logger.info("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Get input size from data
    input_size = X_train.shape[2]
    logger.info(f"Input size: {input_size} features")
    
    # Prepare dataloaders
    logger.info("\nPreparing dataloaders...")
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, config.BATCH_SIZE
    )
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(input_size, config)
    
    # Calculate class weights for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos]).to(config.DEVICE)
    logger.info(f"Class balance - UP: {n_pos} ({n_pos/len(y_train)*100:.1f}%), DOWN: {n_neg}")
    logger.info(f"Using pos_weight: {pos_weight.item():.3f}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    model_path = os.path.join(config.output_dir, 'best_model.pth')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("TRAINING")
    logger.info("="*60)
    
    best_val_acc = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}] '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.4f}')
        
        # Early stopping
        early_stopping(val_loss, model, model_path)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    logger.info("\nLoading best model...")
    model.load_state_dict(torch.load(model_path))
    
    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    
    # Prepare test data
    X_test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_probabilities = torch.sigmoid(test_output)
        test_predicted = (test_probabilities >= 0.5).float()
        
        # Calculate metrics
        correct = (test_predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)
        accuracy = correct / total
        
        # Calculate precision, recall, F1
        true_positives = ((test_predicted == 1) & (y_test_tensor == 1)).sum().item()
        false_positives = ((test_predicted == 1) & (y_test_tensor == 0)).sum().item()
        false_negatives = ((test_predicted == 0) & (y_test_tensor == 1)).sum().item()
        true_negatives = ((test_predicted == 0) & (y_test_tensor == 0)).sum().item()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save test predictions and actuals
    test_pred_numpy = test_predicted.cpu().numpy().flatten()
    test_actual_numpy = y_test_tensor.cpu().numpy().flatten()
    test_prob_numpy = test_probabilities.cpu().numpy().flatten()
    
    np.save(os.path.join(config.output_dir, 'test_predictions.npy'), test_pred_numpy)
    np.save(os.path.join(config.output_dir, 'test_actuals.npy'), test_actual_numpy)
    logger.info("Test predictions and actuals saved")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'best_val_accuracy': best_val_acc,
        'final_epoch': len(history['train_loss']),
        'config': {
            'market': config.MARKET,
            'use_sentiment': config.USE_SENTIMENT,
            'sequence_length': config.SEQUENCE_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'hidden_size': config.HIDDEN_SIZE,
            'num_layers': config.NUM_LAYERS,
            'input_features': input_size
        }
    }
    
    # Save metrics
    metrics_path = os.path.join(config.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # 1. Training history plot
    plot_training_history(
        history,
        config,
        save_path=os.path.join(config.output_dir, 'training_history.png')
    )
    logger.info("Training history plot saved")
    
    # 2. Classification analysis plot
    create_classification_analysis(
        test_actual_numpy,
        test_pred_numpy,
        test_prob_numpy,
        metrics,
        config,
        save_path=os.path.join(config.output_dir, 'classification_analysis.png')
    )
    logger.info("Classification analysis plot saved")
    
    # Print results
    logger.info("\nTest Set Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Positives:  {true_positives}")
    logger.info(f"  True Negatives:  {true_negatives}")
    logger.info(f"  False Positives: {false_positives}")
    logger.info(f"  False Negatives: {false_negatives}")
    
    logger.info(f"\n✅ Training completed! Results saved to {config.output_dir}")
    
    return metrics

if __name__ == "__main__":
    train_model()