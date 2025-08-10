"""
LSTM model architecture for carbon price direction classification
PyTorch implementation with dynamic input size
Outputs logits for binary classification (down/flat vs up)
"""

import torch
import torch.nn as nn

class CarbonPriceLSTM(nn.Module):
    """
    LSTM model for carbon price direction classification
    Input size is determined dynamically from the data
    Outputs single logit for binary classification with BCEWithLogitsLoss
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output for binary classification with BCEWithLogitsLoss
        
        # Activation
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/He initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.normal_(param)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Batch normalization (skip for single samples)
        if last_hidden.size(0) > 1:
            last_hidden = self.batch_norm(last_hidden)
        
        # Fully connected layers
        x = self.dropout(last_hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logit for BCEWithLogitsLoss
        
        return x.squeeze()  # Shape: (batch_size,) for binary classification