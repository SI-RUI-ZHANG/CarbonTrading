"""
Configuration file for LSTM direction classification model
Predicts whether carbon price will move up or down/flat
Includes dynamic input size detection and experiment tracking
"""

import torch
import os

class Config:
    def __init__(self, market='GDEA', use_sentiment=False):
        """Initialize config with specific market and sentiment option"""
        self.MARKET = market
        self.USE_SENTIMENT = use_sentiment
        
    # Data
    @property
    def DATA_DIR(self):
        """Dynamic data directory based on sentiment usage"""
        if self.USE_SENTIMENT:
            return '../../02_Data_Processed/04_LSTM_Ready_Sentiment'
        return '../../02_Data_Processed/04_LSTM_Ready'
    
    # Fixed output directory structure (no timestamps)
    BASE_OUTPUT_DIR = '../../04_Models/'
    
    @property
    def OUTPUT_DIR(self):
        """Dynamic output directory based on market and sentiment - flat structure"""
        sentiment_suffix = 'sentiment' if self.USE_SENTIMENT else 'base'
        return os.path.join(self.BASE_OUTPUT_DIR, f'daily_{self.MARKET}_{sentiment_suffix}')
    
    # Model Architecture for Classification
    # INPUT_SIZE will be determined dynamically from data
    # Using BCEWithLogitsLoss for binary classification with pos_weight
    HIDDEN_SIZE = 80
    NUM_LAYERS = 2
    DROPOUT = 0.35
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    SHUFFLE_TRAIN_LOADER = True  # Easily toggle shuffle behavior
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    SEED = 42