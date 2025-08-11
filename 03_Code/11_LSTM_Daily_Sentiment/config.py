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
    MARKET = 'GDEA'  # 'HBEA' or 'GDEA'
    DATA_DIR = '../../02_Data_Processed/04_LSTM_Ready_Sentiment'
    
    # Experiment Tracking - Unique folder for each run
    RUN_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MARKET}_LSTM_Sentiment"
    BASE_OUTPUT_DIR = '../../04_Models_Sentiment/'
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
    
    # Model Architecture for Classification
    # INPUT_SIZE will be determined dynamically from data
    # Using BCEWithLogitsLoss for binary classification with pos_weight
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
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