"""
Walk-Forward Cross-Validation for Weekly LSTM Models
Implements in-memory processing without intermediate file storage
Preserves all existing outputs with aggregation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import Tuple, Dict, List
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import create_model

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Implements walk-forward cross-validation for weekly LSTM models
    All data processing happens in memory without intermediate file storage
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.train_window = 180  # weeks (~3.5 years)
        self.val_window = 20     # weeks (~5 months)
        self.test_window = 30    # weeks (~7 months)
        self.step_size = 20      # step forward ~5 months
        self.sequence_length = config.SEQUENCE_LENGTH
        
        # Store results in memory
        self.walk_metrics = []
        self.all_predictions = []
        self.all_actuals = []
        self.all_probabilities = []
        self.best_model_state = None
        self.best_walk_num = -1
        self.best_f1 = -1
        
        # Track training histories for aggregation
        self.all_train_losses = []
        self.all_val_losses = []
        self.all_train_accuracies = []
        self.all_val_accuracies = []
        
    def prepare_sequences(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from dataframe - in memory only
        
        Args:
            data: DataFrame with features and target
            is_training: Whether this is training data (affects target extraction)
            
        Returns:
            X: Feature sequences
            y: Target values
        """
        # Calculate direction from log_return if not present
        if 'direction' not in data.columns:
            data['direction'] = (data['log_return'] > 0).astype(int)
        
        # Extract features and target
        feature_cols = [col for col in data.columns if col not in ['direction', 'target']]
        target_col = 'direction'
        
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.sequence_length + 1):
            X.append(features[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length - 1])
        
        return np.array(X), np.array(y)
    
    def prepare_walk_data(self, df: pd.DataFrame, walk_num: int) -> Tuple:
        """
        Prepare data for a single walk - all in memory
        
        Args:
            df: Full dataset
            walk_num: Walk number (0-indexed)
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        """
        # Calculate indices
        start_idx = self.sequence_length + walk_num * self.step_size
        train_end = start_idx + self.train_window
        val_end = train_end + self.val_window
        test_end = min(val_end + self.test_window, len(df))
        
        # Check if we have enough data
        if test_end > len(df):
            logger.warning(f"Walk {walk_num + 1}: Not enough data for full test window")
        
        # Split data
        train_data = df.iloc[start_idx:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:test_end].copy()
        
        # Log walk details
        logger.info(f"Walk {walk_num + 1} data ranges:")
        logger.info(f"  Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} weeks)")
        logger.info(f"  Val: {val_data.index[0].date()} to {val_data.index[-1].date()} ({len(val_data)} weeks)")
        logger.info(f"  Test: {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} weeks)")
        
        # Create sequences
        X_train, y_train = self.prepare_sequences(train_data)
        X_val, y_val = self.prepare_sequences(val_data)
        X_test, y_test = self.prepare_sequences(test_data)
        
        # Reshape for scaling
        n_features = X_train.shape[-1]
        X_train_2d = X_train.reshape(-1, n_features)
        X_val_2d = X_val.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
        
        logger.info(f"  Sequence shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler
    
    def train_walk(self, walk_num: int, data_tuple: Tuple) -> Dict:
        """
        Train model for one walk period - no intermediate file saves
        
        Args:
            walk_num: Walk number (0-indexed)
            data_tuple: Tuple from prepare_walk_data
            
        Returns:
            Dictionary of metrics for this walk
        """
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = data_tuple
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Create fresh model for this walk
        input_size = X_train.shape[-1]
        model = create_model(input_size, self.config)
        
        # Loss and optimizer (Binary classification with single output)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Training loop
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        logger.info(f"Walk {walk_num + 1}: Starting training...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.config.DEVICE), batch_y.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()  # Shape: (batch_size,)
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.config.DEVICE), batch_y.to(self.config.DEVICE)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y.float())
                    
                    val_loss += loss.item()
                    predicted = (torch.sigmoid(outputs) > 0.5).long()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Early stopping check (in memory only)
            if epoch == 0:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        logger.info(f"Walk {walk_num + 1}: Early stopping at epoch {epoch + 1}")
                        break
        
        # Load best model state
        model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        test_predictions = []
        test_probabilities = []
        test_actuals = []
        
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(self.config.DEVICE)
            outputs = model(X_test_tensor).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).long()
            
            test_predictions = predictions.cpu().numpy()
            test_probabilities = probabilities.cpu().numpy()  # Probability of class 1
            test_actuals = y_test
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        test_accuracy = accuracy_score(test_actuals, test_predictions)
        test_precision = precision_score(test_actuals, test_predictions, zero_division=0)
        test_recall = recall_score(test_actuals, test_predictions, zero_division=0)
        test_f1 = f1_score(test_actuals, test_predictions, zero_division=0)
        
        # Store walk metrics
        walk_result = {
            'walk': walk_num + 1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'best_epoch': epoch + 1 - patience_counter,  # Epoch where best val loss was found
            'total_epochs': epoch + 1,
            'train_start': str(pd.Timestamp.now()),  # Placeholder - should get from data
            'test_end': str(pd.Timestamp.now())      # Placeholder - should get from data
        }
        
        self.walk_metrics.append(walk_result)
        
        # Store predictions for aggregation
        self.all_predictions.extend(test_predictions)
        self.all_actuals.extend(test_actuals)
        self.all_probabilities.extend(test_probabilities)
        
        # Store training history for aggregation
        self.all_train_losses.append(train_losses)
        self.all_val_losses.append(val_losses)
        self.all_train_accuracies.append(train_accuracies)
        self.all_val_accuracies.append(val_accuracies)
        
        # Keep best model
        if test_f1 > self.best_f1:
            self.best_f1 = test_f1
            self.best_walk_num = walk_num
            self.best_model_state = model.state_dict().copy()
        
        logger.info(f"Walk {walk_num + 1}: F1={test_f1:.3f}, Acc={test_accuracy:.3f}")
        
        return walk_result
    
    def run_walk_forward(self, df: pd.DataFrame) -> Dict:
        """
        Run complete walk-forward validation
        
        Args:
            df: Full dataset
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Calculate number of walks
        total_weeks = len(df)
        min_start = self.sequence_length
        total_window = self.train_window + self.val_window + self.test_window
        max_start = total_weeks - total_window
        n_walks = ((max_start - min_start) // self.step_size) + 1
        n_walks = min(n_walks, 14)  # Cap at 14 walks
        
        logger.info(f"Starting walk-forward validation with {n_walks} walks")
        
        # Run each walk
        for walk_num in range(n_walks):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Walk {walk_num + 1}/{n_walks}")
            logger.info('='*60)
            
            try:
                # Prepare data for this walk
                data_tuple = self.prepare_walk_data(df, walk_num)
                
                # Train and evaluate
                walk_result = self.train_walk(walk_num, data_tuple)
                
            except Exception as e:
                logger.error(f"Walk {walk_num + 1} failed: {str(e)}")
                continue
        
        # Calculate aggregated metrics
        aggregated_metrics = self.calculate_aggregated_metrics()
        
        return aggregated_metrics
    
    def calculate_aggregated_metrics(self) -> Dict:
        """
        Calculate mean and std metrics across all walks
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.walk_metrics:
            return {}
        
        # Extract metrics
        f1_scores = [m['test_f1'] for m in self.walk_metrics]
        accuracies = [m['test_accuracy'] for m in self.walk_metrics]
        precisions = [m['test_precision'] for m in self.walk_metrics]
        recalls = [m['test_recall'] for m in self.walk_metrics]
        
        # Calculate aggregated metrics
        aggregated = {
            'test_f1': np.mean(f1_scores),
            'test_f1_std': np.std(f1_scores),
            'test_accuracy': np.mean(accuracies),
            'test_accuracy_std': np.std(accuracies),
            'test_precision': np.mean(precisions),
            'test_precision_std': np.std(precisions),
            'test_recall': np.mean(recalls),
            'test_recall_std': np.std(recalls),
            'best_walk': self.best_walk_num + 1,
            'best_f1': self.best_f1,
            'n_walks': len(self.walk_metrics),
            'total_test_samples': len(self.all_predictions)
        }
        
        # Add confusion matrix values
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.all_actuals, self.all_predictions)
        if cm.shape == (2, 2):
            aggregated['true_positives'] = int(cm[1, 1])
            aggregated['true_negatives'] = int(cm[0, 0])
            aggregated['false_positives'] = int(cm[0, 1])
            aggregated['false_negatives'] = int(cm[1, 0])
        
        return aggregated
    
    def create_walk_performance_plot(self, save_path: str):
        """
        Create line plot showing F1 evolution across walks
        
        Args:
            save_path: Path to save the plot
        """
        if not self.walk_metrics:
            logger.warning("No walk metrics to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract data
        walks = [m['walk'] for m in self.walk_metrics]
        f1_scores = [m['test_f1'] for m in self.walk_metrics]
        accuracies = [m['test_accuracy'] for m in self.walk_metrics]
        
        # Top plot: F1 scores and accuracy across walks
        ax1.plot(walks, f1_scores, 'o-', linewidth=2, markersize=8, label='F1 Score', color='blue')
        ax1.plot(walks, accuracies, 's-', linewidth=2, markersize=6, label='Accuracy', color='green', alpha=0.7)
        
        # Add mean lines
        ax1.axhline(np.mean(f1_scores), color='blue', linestyle='--', alpha=0.5,
                   label=f'Mean F1: {np.mean(f1_scores):.3f}')
        ax1.axhline(np.mean(accuracies), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean Acc: {np.mean(accuracies):.3f}')
        
        # Add standard deviation shading for F1
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        ax1.fill_between(walks, mean_f1 - std_f1, mean_f1 + std_f1, 
                         alpha=0.2, color='blue', label=f'±1 STD: {std_f1:.3f}')
        
        ax1.set_xlabel('Walk Number')
        ax1.set_ylabel('Score')
        ax1.set_title(f'{self.config.MARKET} Weekly {"Sentiment" if self.config.USE_SENTIMENT else "Base"} - Walk-Forward Performance')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylim([0, 1])
        
        # Bottom plot: F1 score with best/worst walks highlighted
        ax2.plot(walks, f1_scores, 'o-', linewidth=2, markersize=8, color='blue')
        
        # Highlight best and worst walks
        best_idx = np.argmax(f1_scores)
        worst_idx = np.argmin(f1_scores)
        ax2.plot(walks[best_idx], f1_scores[best_idx], 'o', markersize=12, 
                color='green', label=f'Best: Walk {walks[best_idx]} ({f1_scores[best_idx]:.3f})')
        ax2.plot(walks[worst_idx], f1_scores[worst_idx], 'o', markersize=12,
                color='red', label=f'Worst: Walk {walks[worst_idx]} ({f1_scores[worst_idx]:.3f})')
        
        # Add trend line
        z = np.polyfit(walks, f1_scores, 1)
        p = np.poly1d(z)
        trend_label = 'Improving' if z[0] > 0.001 else ('Declining' if z[0] < -0.001 else 'Stable')
        ax2.plot(walks, p(walks), "r--", alpha=0.5, 
                label=f'Trend: {trend_label} ({z[0]:.4f}/walk)')
        
        ax2.set_xlabel('Walk Number')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Trend Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Walk-forward performance plot saved to {save_path}")
    
    def save_aggregated_outputs(self, output_dir: str):
        """
        Save all aggregated outputs to preserve existing file structure
        
        Args:
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save aggregated metrics.json
        aggregated_metrics = self.calculate_aggregated_metrics()
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        
        # 2. Save walk_forward_metrics.json with detailed results
        detailed_results = {
            'walks': self.walk_metrics,
            'aggregated': aggregated_metrics,
            'config': {
                'train_window': self.train_window,
                'val_window': self.val_window,
                'test_window': self.test_window,
                'step_size': self.step_size,
                'n_walks': len(self.walk_metrics)
            }
        }
        with open(os.path.join(output_dir, 'walk_forward_metrics.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # 3. Save best model
        if self.best_model_state is not None:
            torch.save(self.best_model_state, os.path.join(output_dir, 'best_model.pth'))
            logger.info(f"Saved best model from walk {self.best_walk_num + 1} (F1: {self.best_f1:.3f})")
        
        # 4. Save aggregated predictions and actuals
        np.save(os.path.join(output_dir, 'test_predictions.npy'), np.array(self.all_predictions))
        np.save(os.path.join(output_dir, 'test_actuals.npy'), np.array(self.all_actuals))
        np.save(os.path.join(output_dir, 'test_probabilities.npy'), np.array(self.all_probabilities))
        
        # 5. Create aggregated training history plot
        self.create_aggregated_training_plot(output_dir)
        
        # 6. Create aggregated classification analysis
        self.create_aggregated_classification_analysis(output_dir)
        
        # 7. Create walk performance plot
        self.create_walk_performance_plot(os.path.join(output_dir, 'walk_forward_performance.png'))
        
        logger.info(f"All aggregated outputs saved to {output_dir}")
    
    def create_aggregated_training_plot(self, output_dir: str):
        """
        Create aggregated training history plot
        """
        if not self.all_train_losses:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate mean training curves
        max_epochs = max(len(losses) for losses in self.all_train_losses)
        
        # Pad shorter sequences with NaN
        padded_train_losses = []
        padded_val_losses = []
        for train_loss, val_loss in zip(self.all_train_losses, self.all_val_losses):
            padded_train = list(train_loss) + [np.nan] * (max_epochs - len(train_loss))
            padded_val = list(val_loss) + [np.nan] * (max_epochs - len(val_loss))
            padded_train_losses.append(padded_train)
            padded_val_losses.append(padded_val)
        
        # Calculate means ignoring NaN
        mean_train_loss = np.nanmean(padded_train_losses, axis=0)
        mean_val_loss = np.nanmean(padded_val_losses, axis=0)
        
        # Plot loss
        epochs = range(1, len(mean_train_loss) + 1)
        ax1.plot(epochs, mean_train_loss, label='Train Loss (mean)', color='blue')
        ax1.plot(epochs, mean_val_loss, label='Val Loss (mean)', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Aggregated Training Loss (Mean across walks)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        padded_train_acc = []
        padded_val_acc = []
        for train_acc, val_acc in zip(self.all_train_accuracies, self.all_val_accuracies):
            padded_train = list(train_acc) + [np.nan] * (max_epochs - len(train_acc))
            padded_val = list(val_acc) + [np.nan] * (max_epochs - len(val_acc))
            padded_train_acc.append(padded_train)
            padded_val_acc.append(padded_val)
        
        mean_train_acc = np.nanmean(padded_train_acc, axis=0)
        mean_val_acc = np.nanmean(padded_val_acc, axis=0)
        
        ax2.plot(epochs, mean_train_acc, label='Train Acc (mean)', color='blue')
        ax2.plot(epochs, mean_val_acc, label='Val Acc (mean)', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Aggregated Training Accuracy (Mean across walks)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_aggregated_classification_analysis(self, output_dir: str):
        """
        Create aggregated classification analysis plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.all_actuals, self.all_predictions)
        
        # Calculate aggregated metrics
        aggregated_metrics = self.calculate_aggregated_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Aggregated Confusion Matrix (All Walks)')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Metrics comparison
        metrics_data = {
            'Accuracy': aggregated_metrics['test_accuracy'],
            'Precision': aggregated_metrics['test_precision'],
            'Recall': aggregated_metrics['test_recall'],
            'F1 Score': aggregated_metrics['test_f1']
        }
        bars = axes[0, 1].bar(metrics_data.keys(), metrics_data.values(), 
                             color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0, 1].set_title('Aggregated Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Probability distribution
        axes[1, 0].hist(self.all_probabilities, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[1, 0].set_xlabel('Predicted Probability (Class 1)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Aggregated Prediction Confidence Distribution')
        axes[1, 0].legend()
        
        # 4. Walk-by-walk F1 scores
        if self.walk_metrics:
            walks = [m['walk'] for m in self.walk_metrics]
            f1_scores = [m['test_f1'] for m in self.walk_metrics]
            axes[1, 1].bar(walks, f1_scores, color='steelblue', alpha=0.7)
            axes[1, 1].axhline(np.mean(f1_scores), color='red', linestyle='--',
                              label=f'Mean: {np.mean(f1_scores):.3f}')
            axes[1, 1].set_xlabel('Walk Number')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('F1 Score by Walk')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Add main title
        sentiment_tag = 'Sentiment' if self.config.USE_SENTIMENT else 'Base'
        plt.suptitle(f'{self.config.MARKET} Weekly {sentiment_tag} - Walk-Forward Classification Analysis',
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_analysis.png'), dpi=100, bbox_inches='tight')
        plt.close()