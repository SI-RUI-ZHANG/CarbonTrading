"""
Meta-Model implementations for predicting LSTM reliability
Includes XGBoost, Neural Network, and Logistic Regression options
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class XGBoostMetaModel:
    """XGBoost-based meta-model for predicting LSTM reliability"""
    
    def __init__(self, config):
        """
        Initialize XGBoost meta-model
        
        Args:
            config: MetaConfig object
        """
        self.config = config
        self.model = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        logger.info("Training XGBoost meta-model...")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up parameters
        params = self.config.XGBOOST_PARAMS.copy()
        
        # Training with early stopping
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evallist,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        logger.info("XGBoost training completed")
        
    def predict(self, X):
        """Make predictions"""
        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)
        predictions = (probabilities > 0.5).astype(int)
        return predictions, probabilities
    
    def evaluate(self, X, y, dataset_name="Test"):
        """
        Evaluate model performance
        
        Args:
            X, y: Data to evaluate
            dataset_name: Name for logging
        """
        predictions, probabilities = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'auc_roc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0
        }
        
        logger.info(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric:12s}: {value:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
        logger.info(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
        
        return metrics, predictions, probabilities
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance"""
        if not self.feature_importance:
            logger.warning("No feature importance available")
            return
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Create feature names mapping
        feature_names = self.config.SENTIMENT_FEATURES + ['lstm_pred', 'day_of_week']
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = []
        importances = []
        for feat, imp in sorted_features[:10]:  # Top 10 features
            # Extract feature index from XGBoost feature name (e.g., 'f0' -> 0)
            feat_idx = int(feat[1:]) if feat.startswith('f') else 0
            if feat_idx < len(feature_names):
                features.append(feature_names[feat_idx])
            else:
                features.append(feat)
            importances.append(imp)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title('XGBoost Meta-Model Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
    def save(self, path):
        """Save model"""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")


class NeuralNetMetaModel(nn.Module):
    """Neural network meta-model"""
    
    def __init__(self, input_size, config):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            config: MetaConfig object
        """
        super(NeuralNetMetaModel, self).__init__()
        self.config = config
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in config.NN_PARAMS['hidden_sizes']:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.NN_PARAMS['dropout']))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass"""
        return torch.sigmoid(self.model(x))
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the neural network"""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.parameters(), 
                              lr=self.config.NN_PARAMS['learning_rate'])
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.NN_PARAMS['epochs']):
            # Training
            self.train()
            optimizer.zero_grad()
            
            outputs = self(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.state_dict(), 'best_nn_meta_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.NN_PARAMS['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.load_state_dict(torch.load('best_nn_meta_model.pth'))


class LogisticMetaModel:
    """Logistic regression baseline meta-model"""
    
    def __init__(self, config):
        """Initialize logistic regression model"""
        self.config = config
        self.model = LogisticRegression(
            random_state=config.SEED,
            max_iter=1000,
            class_weight='balanced'
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train logistic regression"""
        logger.info("Training Logistic Regression meta-model...")
        self.model.fit(X_train, y_train)
        
        # Get coefficients for interpretation
        self.coefficients = self.model.coef_[0]
        logger.info("Logistic Regression training completed")
        
    def predict(self, X):
        """Make predictions"""
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        return predictions, probabilities
    
    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        predictions, probabilities = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'auc_roc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0
        }
        
        logger.info(f"\n{dataset_name} Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric:12s}: {value:.3f}")
        
        return metrics, predictions, probabilities
    
    def save(self, path):
        """Save model"""
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")


def create_meta_model(model_type, config, input_size=None):
    """
    Factory function to create meta-model
    
    Args:
        model_type: 'xgboost', 'neural_net', or 'logistic'
        config: MetaConfig object
        input_size: Number of features (for neural net)
    
    Returns:
        Meta-model instance
    """
    if model_type == 'xgboost':
        return XGBoostMetaModel(config)
    elif model_type == 'neural_net':
        if input_size is None:
            raise ValueError("input_size required for neural network")
        return NeuralNetMetaModel(input_size, config)
    elif model_type == 'logistic':
        return LogisticMetaModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")