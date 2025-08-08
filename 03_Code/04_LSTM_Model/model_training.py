"""
Main training script for LSTM carbon price direction classification model
Implements dynamic input size detection and experiment tracking
Uses CrossEntropyLoss for binary classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
import logging

from config import Config
from model_architecture import CarbonPriceLSTM
from utils import load_data, create_dataloaders, EarlyStopping, plot_training_history, save_config
from evaluate import evaluate_model, plot_predictions, calculate_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        y_batch = y_batch.long().to(config.DEVICE)  # Convert to long for classification
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Calculate accuracy
        _, predicted = torch.max(predictions.data, 1)
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
            y_batch = y_batch.long().to(config.DEVICE)  # Convert to long for classification
            
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy

def train_model(config):
    """Main training function"""
    logger.info("="*60)
    logger.info(f"Starting training for {config.MARKET}")
    logger.info(f"Run name: {config.RUN_NAME}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info("="*60)
    
    # Set seed
    set_seed(config.SEED)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(config)
    
    # DYNAMIC INPUT SIZE DETECTION
    config.INPUT_SIZE = X_train.shape[2]
    logger.info(f"Detected input size: {config.INPUT_SIZE} features")
    
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
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer for classification
    criterion = nn.CrossEntropyLoss()
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
    
    logger.info("Starting training...")
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
                'dropout': config.DROPOUT,
                'num_classes': config.NUM_CLASSES
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
    
    return model, metrics

def main():
    """Main execution"""
    config = Config()
    
    # Train model
    model, metrics = train_model(config)
    
    # Return metrics for comparison
    return metrics

if __name__ == "__main__":
    main()