"""
LSTM Model for Weekly Price Direction Prediction
Shared architecture for both sentiment and non-sentiment versions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class WeeklyLSTM(nn.Module):
    """
    LSTM model for weekly carbon price direction prediction
    Architecture is the same as daily model but adapted for weekly data
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(WeeklyLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)  # Binary classification (single output with sigmoid)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) with logits for binary classification
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for classification
        # h_n shape: (num_layers, batch_size, hidden_size)
        # We want the last layer's output
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Pass through fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch_size, 1)
        
        # Note: We return logits, not probabilities
        # BCEWithLogitsLoss will apply sigmoid internally
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Probability of UP movement (after sigmoid)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.predict_proba(x)
        predictions = (probs >= threshold).float()
        return predictions


def create_model(input_size: int, config) -> WeeklyLSTM:
    """
    Factory function to create model with config parameters
    
    Args:
        input_size: Number of input features
        config: Configuration object
        
    Returns:
        Initialized WeeklyLSTM model
    """
    model = WeeklyLSTM(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    # Move to device
    model = model.to(config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Number of layers: {config.NUM_LAYERS}")
    print(f"Dropout: {config.DROPOUT}")
    print(f"Device: {config.DEVICE}")
    
    return model


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 15, delta: float = 0.0001):
        """
        Args:
            patience: How many epochs to wait after last improvement
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float, model: nn.Module, path: str) -> None:
        """
        Check if validation loss improved and save model if it did
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            path: Path to save model
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str) -> None:
        """Save model when validation loss decreases"""
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.best_loss = val_loss