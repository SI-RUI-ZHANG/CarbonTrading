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
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    DROPOUT = 0.30
    
    # Training
    BATCH_SIZE = 64  # Increased for better GPU utilization on M4 Max
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    SHUFFLE_TRAIN_LOADER = True  # Easily toggle shuffle behavior
    
    # DataLoader optimization for Apple Silicon
    NUM_WORKERS = 10  # Parallel data loading (tune based on performance)
    PIN_MEMORY = False  # Set to True if using CUDA, False for MPS
    PREFETCH_FACTOR = 4  # Number of batches to prefetch per worker
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs
    
    # Mixed Precision Training
    USE_AMP = True  # Automatic Mixed Precision for faster training
    
    # Device configuration with Apple Silicon support
    @property
    def DEVICE(self):
        """Smart device selection: MPS (Apple Silicon) > CUDA > CPU"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device
    
    # Memory optimization for M4 Max (36GB unified memory)
    GRADIENT_ACCUMULATION_STEPS = 1  # Increase if running out of memory
    MAX_GRAD_NORM = 1.0  # Gradient clipping value
    
    # Parallel processing for walk-forward validation
    MAX_PARALLEL_WALKS = 3  # Number of walks to run in parallel (tune based on memory)
    
    # Reproducibility
    SEED = 42