"""
LSTM Trading Strategy Classes
Converts LSTM predictions to trading signals for backtesting
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# Type aliases
SignalSeries = pd.Series
NavSeries = pd.Series


class LSTMStrategyBase(ABC):
    """Base class for LSTM-based trading strategies"""
    
    def __init__(self, model_dir: str, market: str, model_type: str):
        """
        Initialize LSTM strategy
        
        Args:
            model_dir: Path to model directory (e.g., '../../04_Models/daily_GDEA_base')
            market: Market name (GDEA or HBEA)
            model_type: Model type (base, sentiment, meta)
        """
        self.model_dir = model_dir
        self.market = market
        self.model_type = model_type
        
        # Load predictions and dates
        self.predictions = None
        self.probabilities = None
        self.dates = None
        self.actuals = None
        self._load_model_outputs()
    
    def _load_model_outputs(self):
        """Load model predictions, probabilities, and dates"""
        try:
            # Load predictions
            pred_path = os.path.join(self.model_dir, 'test_predictions.npy')
            self.predictions = np.load(pred_path)
            
            # Load probabilities if available
            prob_path = os.path.join(self.model_dir, 'test_probabilities.npy')
            if os.path.exists(prob_path):
                self.probabilities = np.load(prob_path)
            
            # Load dates
            dates_path = os.path.join(self.model_dir, 'test_dates.npy')
            if os.path.exists(dates_path):
                self.dates = pd.to_datetime(np.load(dates_path, allow_pickle=True))
            
            # Load actuals for evaluation
            actuals_path = os.path.join(self.model_dir, 'test_actuals.npy')
            if os.path.exists(actuals_path):
                self.actuals = np.load(actuals_path)
                
            print(f"Loaded {len(self.predictions)} predictions from {self.model_dir}")
            
        except Exception as e:
            print(f"Error loading model outputs: {e}")
            raise
    
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate trading signals from predictions"""
        pass
    
    def get_signals_with_dates(self) -> pd.Series:
        """Get signals with proper date index"""
        signals = self.generate_signals()
        if self.dates is not None:
            # Handle length mismatch - use minimum length
            min_len = min(len(signals), len(self.dates))
            if min_len < len(signals):
                print(f"Warning: Truncating {len(signals)} signals to match {len(self.dates)} dates")
                signals = signals[:min_len]
            if min_len < len(self.dates):
                self.dates = self.dates[:min_len]
            signals.index = self.dates
        return signals


class LSTMBinaryStrategy(LSTMStrategyBase):
    """
    Simple binary strategy: Long when prediction is 1, cash when 0
    """
    
    def generate_signals(self) -> pd.Series:
        """Convert binary predictions to trading signals"""
        # LSTM predictions are already 0/1
        signals = pd.Series(self.predictions, name='signal')
        return signals


class LSTMProbabilityStrategy(LSTMStrategyBase):
    """
    Probability-weighted strategy: Position size based on confidence
    """
    
    def __init__(self, model_dir: str, market: str, model_type: str,
                 threshold: float = 0.5, scale_position: bool = True):
        """
        Initialize probability strategy
        
        Args:
            model_dir: Path to model directory
            market: Market name
            model_type: Model type
            threshold: Minimum probability to enter position
            scale_position: If True, scale position by probability
        """
        super().__init__(model_dir, market, model_type)
        self.threshold = threshold
        self.scale_position = scale_position
    
    def generate_signals(self) -> pd.Series:
        """Generate signals with position sizing based on probability"""
        if self.probabilities is None:
            # Fallback to binary if no probabilities
            return pd.Series(self.predictions, name='signal')
        
        if self.scale_position:
            # Scale position size by probability distance from 0.5
            # prob=0.5 -> position=0, prob=1.0 -> position=1
            confidence = np.abs(self.probabilities - 0.5) * 2
            signals = np.where(self.probabilities > self.threshold,
                             confidence,  # Long with scaled position
                             0)          # No position
        else:
            # Binary decision based on threshold
            signals = (self.probabilities > self.threshold).astype(int)
        
        return pd.Series(signals, name='signal')


class LSTMConfidenceStrategy(LSTMStrategyBase):
    """
    High-confidence strategy: Only trade when model is very confident
    """
    
    def __init__(self, model_dir: str, market: str, model_type: str,
                 high_conf: float = 0.7, low_conf: float = 0.3):
        """
        Initialize confidence strategy
        
        Args:
            model_dir: Path to model directory
            market: Market name
            model_type: Model type
            high_conf: Threshold for high confidence long
            low_conf: Threshold for high confidence short (not used for long-only)
        """
        super().__init__(model_dir, market, model_type)
        self.high_conf = high_conf
        self.low_conf = low_conf
    
    def generate_signals(self) -> pd.Series:
        """Generate signals only for high confidence predictions"""
        if self.probabilities is None:
            # Fallback to binary if no probabilities
            return pd.Series(self.predictions, name='signal')
        
        # Only go long when probability > high_conf
        # Stay in cash otherwise (long-only strategy)
        signals = (self.probabilities > self.high_conf).astype(int)
        
        return pd.Series(signals, name='signal')


class MetaModelStrategy:
    """
    Meta model strategy: Uses error reversal predictions
    Always trades (100% coverage)
    Note: Does not inherit from LSTMStrategyBase due to different file structure
    """
    
    def __init__(self, model_dir: str, market: str):
        """
        Initialize meta model strategy
        
        Args:
            model_dir: Path to meta model directory
            market: Market name
        """
        self.model_dir = model_dir
        self.market = market
        self.model_type = 'meta'
        self.predictions = None
        self.dates = None
        
        # Load meta-specific predictions
        self._load_meta_predictions()
    
    def _load_meta_predictions(self):
        """Load meta model specific predictions"""
        # Check for validation predictions (meta models save differently)
        val_pred_path = os.path.join(self.model_dir, 'val_predictions.npy')
        if os.path.exists(val_pred_path):
            # Meta models save validation predictions
            self.predictions = np.load(val_pred_path)
            print(f"Loaded {len(self.predictions)} meta predictions")
            
            # Try to load dates from val_dates.npy
            val_dates_path = os.path.join(self.model_dir, 'val_dates.npy')
            if os.path.exists(val_dates_path):
                self.dates = pd.to_datetime(np.load(val_dates_path, allow_pickle=True))
    
    def generate_signals(self) -> pd.Series:
        """Generate signals from meta model predictions"""
        # Meta model predictions are already processed
        if self.predictions is not None:
            signals = pd.Series(self.predictions, name='signal')
        else:
            # Return empty series if no predictions
            signals = pd.Series([], name='signal')
        return signals
    
    def get_signals_with_dates(self) -> pd.Series:
        """Get signals with proper date index"""
        signals = self.generate_signals()
        if self.dates is not None and len(signals) > 0:
            signals.index = self.dates
        return signals


class EnsembleStrategy(LSTMStrategyBase):
    """
    Ensemble strategy: Combines multiple model predictions
    """
    
    def __init__(self, model_dirs: list, market: str, voting: str = 'majority'):
        """
        Initialize ensemble strategy
        
        Args:
            model_dirs: List of model directories to ensemble
            market: Market name
            voting: Voting method ('majority', 'average', 'weighted')
        """
        self.model_dirs = model_dirs
        self.market = market
        self.voting = voting
        self.all_predictions = []
        self.all_probabilities = []
        
        # Load all model predictions
        for model_dir in model_dirs:
            try:
                preds = np.load(os.path.join(model_dir, 'test_predictions.npy'))
                self.all_predictions.append(preds)
                
                prob_path = os.path.join(model_dir, 'test_probabilities.npy')
                if os.path.exists(prob_path):
                    probs = np.load(prob_path)
                    self.all_probabilities.append(probs)
            except:
                print(f"Could not load predictions from {model_dir}")
        
        # Use first model for dates
        dates_path = os.path.join(model_dirs[0], 'test_dates.npy')
        if os.path.exists(dates_path):
            self.dates = pd.to_datetime(np.load(dates_path, allow_pickle=True))
    
    def generate_signals(self) -> pd.Series:
        """Generate ensemble signals"""
        if self.voting == 'majority':
            # Majority voting
            ensemble_preds = np.mean(self.all_predictions, axis=0)
            signals = (ensemble_preds > 0.5).astype(int)
        
        elif self.voting == 'average' and self.all_probabilities:
            # Average probabilities
            ensemble_probs = np.mean(self.all_probabilities, axis=0)
            signals = (ensemble_probs > 0.5).astype(int)
        
        else:
            # Default to majority
            ensemble_preds = np.mean(self.all_predictions, axis=0)
            signals = (ensemble_preds > 0.5).astype(int)
        
        return pd.Series(signals, name='signal')