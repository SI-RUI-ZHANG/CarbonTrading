"""
Utility functions for LSTM model training
Includes data loading, dataset creation, and visualization
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
from datetime import datetime

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(config):
    """Load preprocessed numpy arrays"""
    market = config.MARKET
    data_dir = config.DATA_DIR
    
    X_train = np.load(f'{data_dir}/{market}_X_train.npy')
    y_train = np.load(f'{data_dir}/{market}_y_train.npy')
    X_val = np.load(f'{data_dir}/{market}_X_val.npy')
    y_val = np.load(f'{data_dir}/{market}_y_val.npy')
    X_test = np.load(f'{data_dir}/{market}_X_test.npy')
    y_test = np.load(f'{data_dir}/{market}_y_test.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Create PyTorch DataLoaders with configurable shuffle"""
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Use shuffle setting from config for training loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=config.SHUFFLE_TRAIN_LOADER  # Now configurable
    )
    
    # Validation and test loaders should never shuffle
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, config):
    """Plot training history for classification"""
    plt.figure(figsize=(15, 5))
    
    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title(f'{config.MARKET} - Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy subplot
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{config.MARKET} - Accuracy History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed accuracy (last 2/3)
    plt.subplot(1, 3, 3)
    start = len(train_accuracies) // 3
    plt.plot(range(start, len(train_accuracies)), train_accuracies[start:], label='Training Accuracy')
    plt.plot(range(start, len(val_accuracies)), val_accuracies[start:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{config.MARKET} - Accuracy (Last 2/3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/training_history.png', dpi=100)
    plt.show()

def save_config(config):
    """Save configuration to JSON for reproducibility"""
    config_dict = {
        'MARKET': config.MARKET,
        'RUN_NAME': config.RUN_NAME,
        'TASK_TYPE': 'Classification',
        'NUM_CLASSES': config.NUM_CLASSES,
        'INPUT_SIZE': getattr(config, 'INPUT_SIZE', 'Determined from data'),
        'HIDDEN_SIZE': config.HIDDEN_SIZE,
        'NUM_LAYERS': config.NUM_LAYERS,
        'DROPOUT': config.DROPOUT,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'EARLY_STOPPING_PATIENCE': config.EARLY_STOPPING_PATIENCE,
        'SHUFFLE_TRAIN_LOADER': config.SHUFFLE_TRAIN_LOADER,
        'SEED': config.SEED,
        'DEVICE': str(config.DEVICE)
    }
    
    with open(f'{config.OUTPUT_DIR}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)