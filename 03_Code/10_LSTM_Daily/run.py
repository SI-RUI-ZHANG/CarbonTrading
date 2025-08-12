"""
LSTM Daily Model with Walk-Forward Validation
Supports both base and sentiment features for GDEA and HBEA markets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse
import sys

# Add parent directory to path for utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from model_architecture import CarbonPriceLSTM
from utils import create_dataloaders, save_config, TimeSeriesDataset
from evaluate import evaluate_model
from torch.utils.data import DataLoader

# Import performance tracker
try:
    from utils.performance_tracker import update_performance_summary
except ImportError:
    # Fallback if utils is not a package
    sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
    from performance_tracker import update_performance_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """Walk-forward cross-validation for daily LSTM"""
    
    def __init__(self, config: Config):
        self.config = config
        self.train_window = 700  # days
        self.val_window = 150    # days  
        self.test_window = 200   # days
        self.step_size = 150     # days
        self.sequence_length = 60  # 60-day lookback
        
        # Store predictions from all walks for aggregation
        self.all_predictions = []
        self.all_actuals = []
        self.all_probabilities = []
        self.all_dates = []  # Store dates for backtesting alignment
        
    def load_data(self) -> pd.DataFrame:
        """Load the appropriate dataset based on sentiment usage"""
        if self.config.USE_SENTIMENT:
            file_path = f'../../02_Data_Processed/10_Sentiment_Final_Merged/{self.config.MARKET}_LSTM_with_sentiment.parquet'
        else:
            file_path = f'../../02_Data_Processed/03_Feature_Engineered/{self.config.MARKET}_LSTM_advanced.parquet'
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Drop NaN rows
        df_clean = df.dropna()
        logger.info(f"Data shape after cleaning: {df_clean.shape}")
        
        return df_clean
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def prepare_walk_data(self, df: pd.DataFrame, start_idx: int) -> Tuple:
        """Prepare data for a single walk"""
        # Define windows
        train_end = start_idx + self.train_window
        val_end = train_end + self.val_window
        test_end = val_end + self.test_window
        
        # Check if we have enough data
        if test_end > len(df):
            return None
        
        # Split data
        train_df = df.iloc[start_idx:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:test_end]
        
        # Identify features (exclude log_return which is target source)
        feature_cols = [col for col in df.columns if col != 'log_return']
        
        # Scale features (fit on train only)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_features = scaler.fit_transform(train_df[feature_cols])
        val_features = scaler.transform(val_df[feature_cols])
        test_features = scaler.transform(test_df[feature_cols])
        
        # Create direction labels (1 for up, 0 for down/flat)
        train_targets = (train_df['log_return'].values > 0).astype(np.float32)
        val_targets = (val_df['log_return'].values > 0).astype(np.float32)
        test_targets = (test_df['log_return'].values > 0).astype(np.float32)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_features, train_targets)
        X_val, y_val = self.create_sequences(val_features, val_targets)
        X_test, y_test = self.create_sequences(test_features, test_targets)
        
        # Get test dates (after sequence creation, so we get the right dates)
        test_dates = test_df.index[self.sequence_length:].to_numpy()
        
        return X_train, y_train, X_val, y_val, X_test, y_test, test_dates
    
    def train_walk(self, X_train, y_train, X_val, y_val) -> CarbonPriceLSTM:
        """Train model for one walk"""
        # Detect input size
        input_size = X_train.shape[2]
        
        # Create model
        model = CarbonPriceLSTM(
            input_size=input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        ).to(self.config.DEVICE)
        
        # Calculate pos_weight for class imbalance
        num_negatives = np.sum(y_train == 0)
        num_positives = np.sum(y_train == 1)
        pos_weight = torch.tensor([num_negatives / num_positives], 
                                 dtype=torch.float32, device=self.config.DEVICE)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            X_train, y_train, X_val, y_val, X_val, y_val, self.config
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.config.DEVICE)
                y_batch = y_batch.float().to(self.config.DEVICE)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.config.DEVICE)
                    y_batch = y_batch.float().to(self.config.DEVICE)
                    
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model
    
    def evaluate_walk(self, model, X_test, y_test) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on test set"""
        model.eval()
        
        # Create test loader
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Get predictions
        predictions, actuals, metrics, cm, probabilities = evaluate_model(model, test_loader, self.config)
        
        return metrics, predictions, actuals, probabilities
    
    def run(self) -> Dict:
        """Run complete walk-forward validation"""
        logger.info("="*80)
        logger.info(f"WALK-FORWARD VALIDATION - {self.config.MARKET} "
                   f"({'With' if self.config.USE_SENTIMENT else 'Without'} Sentiment)")
        logger.info("="*80)
        
        # Load data
        df = self.load_data()
        
        # Calculate number of walks (targeting 10 walks for extended coverage)
        total_samples = len(df)
        min_required = self.train_window + self.val_window + self.test_window + self.sequence_length
        max_walks = (total_samples - min_required) // self.step_size + 1
        n_walks = min(10, max_walks)  # Use 10 walks for extended coverage to May 2024
        
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Windows: train={self.train_window}, val={self.val_window}, test={self.test_window}")
        logger.info(f"Step size: {self.step_size}, Sequence length: {self.sequence_length}")
        logger.info(f"Number of walks: {n_walks}")
        
        walk_metrics = []
        test_samples_total = 0
        
        for walk_idx in range(n_walks):
            start_idx = walk_idx * self.step_size
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk {walk_idx + 1}/{n_walks}")
            logger.info(f"{'='*60}")
            
            # Prepare data for this walk
            walk_data = self.prepare_walk_data(df, start_idx)
            if walk_data is None:
                logger.warning(f"Not enough data for walk {walk_idx + 1}, stopping")
                break
            
            X_train, y_train, X_val, y_val, X_test, y_test, test_dates = walk_data
            
            # Log walk info
            train_start_date = df.index[start_idx]
            train_end_date = df.index[start_idx + self.train_window - 1]
            test_start_date = df.index[start_idx + self.train_window + self.val_window]
            test_end_date = df.index[min(start_idx + self.train_window + self.val_window + self.test_window - 1, len(df)-1)]
            
            logger.info(f"Train period: {train_start_date.date()} to {train_end_date.date()}")
            logger.info(f"Test period: {test_start_date.date()} to {test_end_date.date()}")
            logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
            
            # Train model
            logger.info("Training model...")
            model = self.train_walk(X_train, y_train, X_val, y_val)
            
            # Evaluate
            logger.info("Evaluating on test set...")
            metrics, test_predictions, test_actuals, test_probabilities = self.evaluate_walk(model, X_test, y_test)
            walk_metrics.append(metrics)
            test_samples_total += len(X_test)
            
            # Store predictions for aggregation
            self.all_predictions.extend(test_predictions)
            self.all_actuals.extend(test_actuals)
            self.all_probabilities.extend(test_probabilities)
            self.all_dates.extend(test_dates)
            
            # Log metrics
            logger.info(f"Walk {walk_idx + 1} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Aggregate metrics
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*80)
        
        aggregated_metrics = {}
        for key in walk_metrics[0].keys():
            if isinstance(walk_metrics[0][key], (int, float)):
                values = [m[key] for m in walk_metrics]
                aggregated_metrics[f'{key}_mean'] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
                aggregated_metrics[f'{key}_min'] = np.min(values)
                aggregated_metrics[f'{key}_max'] = np.max(values)
        
        # Add summary info
        aggregated_metrics['n_walks'] = len(walk_metrics)
        aggregated_metrics['total_test_samples'] = test_samples_total
        aggregated_metrics['market'] = self.config.MARKET
        aggregated_metrics['use_sentiment'] = self.config.USE_SENTIMENT
        
        # Log summary
        logger.info(f"Number of successful walks: {len(walk_metrics)}")
        logger.info(f"Total test samples: {test_samples_total}")
        logger.info(f"\nAggregated Metrics (Mean ± Std):")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mean_val = aggregated_metrics[f'{metric}_mean']
            std_val = aggregated_metrics[f'{metric}_std']
            logger.info(f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Save aggregated metrics
        aggregated_metrics_clean = convert_numpy_types(aggregated_metrics)
        with open(os.path.join(self.config.OUTPUT_DIR, 'walk_forward_metrics.json'), 'w') as f:
            json.dump(aggregated_metrics_clean, f, indent=4)
        
        # Save individual walk metrics
        walk_metrics_clean = convert_numpy_types(walk_metrics)
        with open(os.path.join(self.config.OUTPUT_DIR, 'walk_metrics_all.json'), 'w') as f:
            json.dump(walk_metrics_clean, f, indent=4)
        
        # Save config
        save_config(self.config)
        
        # Save aggregated predictions and actuals (matching weekly format)
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_predictions.npy'), np.array(self.all_predictions))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_actuals.npy'), np.array(self.all_actuals))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_probabilities.npy'), np.array(self.all_probabilities))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_dates.npy'), np.array(self.all_dates))
        
        logger.info(f"\nSaved aggregated predictions: {len(self.all_predictions)} samples with dates")
        logger.info(f"Results saved to: {self.config.OUTPUT_DIR}")
        
        return aggregated_metrics

def main():
    """Main execution - runs all combinations of markets and sentiment by default"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Daily LSTM Model using Walk-Forward Validation')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run (default: both)')
    parser.add_argument('--sentiment', choices=['base', 'sentiment', 'both'],
                       default='both', help='Feature set to use (default: both)')
    args = parser.parse_args()
    
    # Determine which markets to run
    if args.market == 'both':
        markets = ['GDEA', 'HBEA']
    else:
        markets = [args.market]
    
    # Determine which feature sets to use
    sentiment_options = []
    if args.sentiment in ['base', 'both']:
        sentiment_options.append(False)  # Base features
    if args.sentiment in ['sentiment', 'both']:
        sentiment_options.append(True)   # With sentiment
    
    # Track all results
    all_results = []
    
    logger.info("\n" + "="*80)
    logger.info("DAILY LSTM - WALK-FORWARD VALIDATION")
    logger.info(f"Markets: {', '.join(markets)}")
    logger.info(f"Features: {args.sentiment}")
    logger.info("="*80)
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all combinations
    for market in markets:
        for use_sentiment in sentiment_options:
            feature_type = "With Sentiment" if use_sentiment else "Base"
            
            logger.info("\n" + "="*80)
            logger.info(f"RUNNING: {market} - {feature_type}")
            logger.info("="*80)
            
            try:
                # Create config
                config = Config(market=market, use_sentiment=use_sentiment)
                
                # Run walk-forward validation
                validator = WalkForwardValidator(config)
                metrics = validator.run()
                
                # Store results
                result = {
                    'market': market,
                    'use_sentiment': use_sentiment,
                    'feature_type': feature_type,
                    'metrics': metrics,
                    'status': 'success'
                }
                all_results.append(result)
                
                # Update performance summary with aggregated metrics
                # Map walk-forward metrics to performance tracker format
                tracker_metrics = {
                    'test_accuracy': metrics.get('accuracy_mean', 0),
                    'test_precision': metrics.get('precision_mean', 0),
                    'test_recall': metrics.get('recall_mean', 0),
                    'test_f1': metrics.get('f1_score_mean', 0)
                }
                
                update_performance_summary(
                    model_type=f'daily_{"sentiment" if use_sentiment else "base"}',
                    market=market,
                    metrics=tracker_metrics,
                    features_type='+Sentiment' if use_sentiment else 'Base',
                    summary_path='../../04_Models/performance_summary.txt'
                )
                
                logger.info(f"✅ {market} - {feature_type} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {market} - {feature_type} failed: {str(e)}")
                result = {
                    'market': market,
                    'use_sentiment': use_sentiment,
                    'feature_type': feature_type,
                    'error': str(e),
                    'status': 'failed'
                }
                all_results.append(result)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    for result in all_results:
        if result['status'] == 'success':
            metrics = result['metrics']
            logger.info(f"{result['market']} - {result['feature_type']}:")
            logger.info(f"  Accuracy: {metrics.get('accuracy_mean', 0):.4f} ± {metrics.get('accuracy_std', 0):.4f}")
            logger.info(f"  F1-Score: {metrics.get('f1_score_mean', 0):.4f} ± {metrics.get('f1_score_std', 0):.4f}")
            logger.info(f"  Walks: {metrics.get('n_walks', 0)}, Test samples: {metrics.get('total_test_samples', 0)}")
        else:
            logger.info(f"{result['market']} - {result['feature_type']}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Count successes
    n_success = sum(1 for r in all_results if r['status'] == 'success')
    n_total = len(all_results)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Completed {n_success}/{n_total} experiments successfully")
    logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    return all_results

if __name__ == "__main__":
    main()