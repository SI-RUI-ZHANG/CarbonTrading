"""
Configuration file for LSTM direction classification model
Predicts whether carbon price will move up or down/flat
Includes dynamic input size detection and experiment tracking
"""

import torch
import os
from datetime import datetime

class Config:
    # Data
    MARKET = 'GDEA'  # or 'HBEA'
    DATA_DIR = '../../02_Data_Processed/04_LSTM_Ready'
    
    # Experiment Tracking - Unique folder for each run
    RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MARKET}_LSTM_Classification"
    BASE_OUTPUT_DIR = '../../04_Models/'
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
    
    # Model Architecture for Classification
    # INPUT_SIZE will be determined dynamically from data
    NUM_CLASSES = 2  # Binary classification: Down/Flat (0) vs Up (1)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    DROPOUT = 0.2
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100  # Quick test to check completion
    EARLY_STOPPING_PATIENCE = 15
    SHUFFLE_TRAIN_LOADER = True  # Easily toggle shuffle behavior
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reproducibility
    SEED = 42