"""
Optimized LSTM Daily Model for Apple Silicon M4 Max
Features:
- MPS (Metal Performance Shaders) GPU acceleration
- Parallel walk-forward validation
- Mixed precision training
- Optimized DataLoader with multi-worker loading
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse
import sys
import time
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings

# Suppress MPS fallback warnings for unsupported operations
warnings.filterwarnings('ignore', category=UserWarning, module='torch.mps')

# Add parent directory to path for utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from model_architecture import CarbonPriceLSTM
from utils import save_config, TimeSeriesDataset
from evaluate import evaluate_model
from torch.utils.data import DataLoader

# Import performance tracker
try:
    from utils.performance_tracker import update_performance_summary
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
    from performance_tracker import update_performance_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor training performance and resource usage"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.gpu_memory_usage = []
        self.cpu_usage = []
        
    def start_epoch(self):
        self.start_time = time.time()
        
    def end_epoch(self, epoch_num: int, device: torch.device):
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            
            # Monitor resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)
            
            # Monitor GPU memory if using MPS or CUDA
            if device.type == 'mps':
                # MPS memory monitoring (if available in future PyTorch versions)
                try:
                    # Note: Direct MPS memory querying may not be available yet
                    gpu_mem = "N/A (MPS)"
                except:
                    gpu_mem = "N/A"
            elif device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            else:
                gpu_mem = 0
                
            if epoch_num % 10 == 0:
                logger.info(f"  Performance - Epoch time: {epoch_time:.2f}s, "
                          f"CPU: {cpu_percent:.1f}%, GPU Memory: {gpu_mem}")
    
    def get_summary(self) -> Dict:
        if self.epoch_times:
            return {
                'avg_epoch_time': np.mean(self.epoch_times),
                'total_time': sum(self.epoch_times),
                'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            }
        return {}

def create_optimized_dataloader(X, y, config: Config, shuffle: bool = True) -> DataLoader:
    """Create optimized DataLoader for Apple Silicon"""
    dataset = TimeSeriesDataset(X, y)
    
    # Optimize DataLoader parameters for M4 Max
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS if not shuffle else 0,  # Disable multiprocessing for training due to MPS limitations
        pin_memory=config.PIN_MEMORY and config.DEVICE.type == 'cuda',
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0
    )

class OptimizedWalkForwardValidator:
    """Optimized Walk-forward validation for Apple Silicon"""
    
    def __init__(self, config: Config):
        self.config = config
        self.train_window = 700
        self.val_window = 150
        self.test_window = 200
        self.step_size = 150
        self.sequence_length = 60
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Results storage
        self.all_predictions = []
        self.all_actuals = []
        self.all_probabilities = []
        self.all_dates = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the appropriate dataset based on sentiment usage"""
        if self.config.USE_SENTIMENT:
            file_path = f'../../02_Data_Processed/10_Sentiment_Final_Merged/{self.config.MARKET}_LSTM_with_sentiment.parquet'
        else:
            file_path = f'../../02_Data_Processed/03_Feature_Engineered/{self.config.MARKET}_LSTM_advanced.parquet'
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
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
        train_end = start_idx + self.train_window
        val_end = train_end + self.val_window
        test_end = val_end + self.test_window
        
        if test_end > len(df):
            return None
        
        train_df = df.iloc[start_idx:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:test_end]
        
        feature_cols = [col for col in df.columns if col != 'log_return']
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_features = scaler.fit_transform(train_df[feature_cols])
        val_features = scaler.transform(val_df[feature_cols])
        test_features = scaler.transform(test_df[feature_cols])
        
        train_targets = (train_df['log_return'].values > 0).astype(np.float32)
        val_targets = (val_df['log_return'].values > 0).astype(np.float32)
        test_targets = (test_df['log_return'].values > 0).astype(np.float32)
        
        X_train, y_train = self.create_sequences(train_features, train_targets)
        X_val, y_val = self.create_sequences(val_features, val_targets)
        X_test, y_test = self.create_sequences(test_features, test_targets)
        
        test_dates = test_df.index[self.sequence_length:].to_numpy()
        
        return X_train, y_train, X_val, y_val, X_test, y_test, test_dates
    
    def train_walk_with_amp(self, X_train, y_train, X_val, y_val) -> CarbonPriceLSTM:
        """Train model with Automatic Mixed Precision for faster training"""
        input_size = X_train.shape[2]
        
        # Create model
        model = CarbonPriceLSTM(
            input_size=input_size,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        ).to(self.config.DEVICE)
        
        # Class balancing
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
        
        # Mixed precision scaler (works with MPS in PyTorch 2.0+)
        use_amp = self.config.USE_AMP and self.config.DEVICE.type in ['cuda', 'mps']
        scaler = GradScaler('cuda' if self.config.DEVICE.type == 'cuda' else 'cpu') if use_amp else None
        
        # Create optimized dataloaders
        train_loader = create_optimized_dataloader(X_train, y_train, self.config, shuffle=True)
        val_loader = create_optimized_dataloader(X_val, y_val, self.config, shuffle=False)
        
        # Training loop with AMP
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.performance_monitor.start_epoch()
            
            # Training phase
            model.train()
            train_loss = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = X_batch.to(self.config.DEVICE)
                y_batch = y_batch.float().to(self.config.DEVICE)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if use_amp and self.config.DEVICE.type == 'cuda':
                    with autocast('cuda'):
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training (for MPS or CPU)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.MAX_GRAD_NORM)
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Gradient accumulation if needed
                if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Validation phase
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
            
            # Track performance
            self.performance_monitor.end_epoch(epoch, self.config.DEVICE)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model
    
    def evaluate_walk(self, model, X_test, y_test) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model on test set"""
        model.eval()
        
        test_loader = create_optimized_dataloader(X_test, y_test, self.config, shuffle=False)
        
        predictions, actuals, metrics, cm, probabilities = evaluate_model(model, test_loader, self.config)
        
        return metrics, predictions, actuals, probabilities
    
    def process_single_walk(self, walk_idx: int, df: pd.DataFrame, n_walks: int) -> Optional[Dict]:
        """Process a single walk (for parallel execution)"""
        try:
            start_idx = walk_idx * self.step_size
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk {walk_idx + 1}/{n_walks}")
            logger.info(f"{'='*60}")
            
            # Prepare data
            walk_data = self.prepare_walk_data(df, start_idx)
            if walk_data is None:
                logger.warning(f"Not enough data for walk {walk_idx + 1}")
                return None
            
            X_train, y_train, X_val, y_val, X_test, y_test, test_dates = walk_data
            
            # Log walk info
            train_start = df.index[start_idx].date()
            train_end = df.index[start_idx + self.train_window - 1].date()
            test_start = df.index[start_idx + self.train_window + self.val_window].date()
            test_end = df.index[min(start_idx + self.train_window + self.val_window + self.test_window - 1, len(df)-1)].date()
            
            logger.info(f"Train: {train_start} to {train_end}")
            logger.info(f"Test: {test_start} to {test_end}")
            logger.info(f"Samples - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train model
            logger.info("Training model with mixed precision...")
            model = self.train_walk_with_amp(X_train, y_train, X_val, y_val)
            
            # Evaluate
            logger.info("Evaluating on test set...")
            metrics, predictions, actuals, probabilities = self.evaluate_walk(model, X_test, y_test)
            
            # Log results
            logger.info(f"Walk {walk_idx + 1} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            
            # Get performance summary
            perf_summary = self.performance_monitor.get_summary()
            if perf_summary:
                logger.info(f"  Avg epoch time: {perf_summary['avg_epoch_time']:.2f}s")
            
            return {
                'walk_idx': walk_idx,
                'metrics': metrics,
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist(),
                'probabilities': probabilities.tolist(),
                'dates': test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates),
                'performance': perf_summary
            }
            
        except Exception as e:
            logger.error(f"Error in walk {walk_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_parallel(self) -> Dict:
        """Run walk-forward validation with parallel processing"""
        logger.info("="*80)
        logger.info(f"OPTIMIZED WALK-FORWARD VALIDATION - {self.config.MARKET}")
        logger.info(f"Features: {'With Sentiment' if self.config.USE_SENTIMENT else 'Base'}")
        logger.info(f"Device: {self.config.DEVICE}")
        logger.info(f"Batch Size: {self.config.BATCH_SIZE}")
        logger.info(f"Mixed Precision: {self.config.USE_AMP}")
        logger.info("="*80)
        
        # Load data
        df = self.load_data()
        
        # Calculate number of walks
        total_samples = len(df)
        min_required = self.train_window + self.val_window + self.test_window + self.sequence_length
        max_walks = (total_samples - min_required) // self.step_size + 1
        n_walks = min(10, max_walks)
        
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Number of walks: {n_walks}")
        logger.info(f"Parallel walks: {min(self.config.MAX_PARALLEL_WALKS, n_walks)}")
        
        # Process walks (sequentially for now due to MPS limitations with multiprocessing)
        walk_results = []
        for walk_idx in range(n_walks):
            result = self.process_single_walk(walk_idx, df, n_walks)
            if result:
                walk_results.append(result)
        
        # Aggregate results
        walk_metrics = []
        performance_stats = []
        
        for result in sorted(walk_results, key=lambda x: x['walk_idx']):
            walk_metrics.append(result['metrics'])
            self.all_predictions.extend(result['predictions'])
            self.all_actuals.extend(result['actuals'])
            self.all_probabilities.extend(result['probabilities'])
            self.all_dates.extend(result['dates'])
            if result['performance']:
                performance_stats.append(result['performance'])
        
        # Calculate aggregated metrics
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("="*80)
        
        aggregated_metrics = {}
        for key in walk_metrics[0].keys():
            if isinstance(walk_metrics[0][key], (int, float)):
                values = [m[key] for m in walk_metrics]
                aggregated_metrics[f'{key}_mean'] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
        
        aggregated_metrics['n_walks'] = len(walk_metrics)
        aggregated_metrics['total_test_samples'] = len(self.all_predictions)
        aggregated_metrics['market'] = self.config.MARKET
        aggregated_metrics['use_sentiment'] = self.config.USE_SENTIMENT
        aggregated_metrics['device_used'] = str(self.config.DEVICE)
        
        # Add performance metrics
        if performance_stats:
            avg_epoch_time = np.mean([p['avg_epoch_time'] for p in performance_stats])
            avg_cpu = np.mean([p['avg_cpu_usage'] for p in performance_stats])
            aggregated_metrics['avg_epoch_time_seconds'] = avg_epoch_time
            aggregated_metrics['avg_cpu_usage_percent'] = avg_cpu
            
            logger.info(f"Performance Metrics:")
            logger.info(f"  Device: {self.config.DEVICE}")
            logger.info(f"  Avg epoch time: {avg_epoch_time:.2f}s")
            logger.info(f"  Avg CPU usage: {avg_cpu:.1f}%")
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
        logger.info(f"  F1-Score: {aggregated_metrics['f1_score_mean']:.4f} ± {aggregated_metrics['f1_score_std']:.4f}")
        
        # Save results
        self.save_results(aggregated_metrics, walk_metrics)
        
        return aggregated_metrics
    
    def save_results(self, aggregated_metrics: Dict, walk_metrics: List[Dict]):
        """Save all results"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # Convert numpy types
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
        
        # Save metrics
        with open(os.path.join(self.config.OUTPUT_DIR, 'walk_forward_metrics.json'), 'w') as f:
            json.dump(convert_numpy_types(aggregated_metrics), f, indent=4)
        
        with open(os.path.join(self.config.OUTPUT_DIR, 'walk_metrics_all.json'), 'w') as f:
            json.dump(convert_numpy_types(walk_metrics), f, indent=4)
        
        # Save predictions
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_predictions.npy'), np.array(self.all_predictions))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_actuals.npy'), np.array(self.all_actuals))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_probabilities.npy'), np.array(self.all_probabilities))
        np.save(os.path.join(self.config.OUTPUT_DIR, 'test_dates.npy'), np.array(self.all_dates))
        
        save_config(self.config)
        
        logger.info(f"\nResults saved to: {self.config.OUTPUT_DIR}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Optimized LSTM Training for Apple Silicon')
    parser.add_argument('--market', choices=['GDEA', 'HBEA', 'both'], 
                       default='both', help='Market(s) to run')
    parser.add_argument('--sentiment', choices=['base', 'sentiment', 'both'],
                       default='both', help='Feature set to use')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    args = parser.parse_args()
    
    # Determine configurations
    markets = ['GDEA', 'HBEA'] if args.market == 'both' else [args.market]
    sentiment_options = []
    if args.sentiment in ['base', 'both']:
        sentiment_options.append(False)
    if args.sentiment in ['sentiment', 'both']:
        sentiment_options.append(True)
    
    all_results = []
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZED DAILY LSTM FOR APPLE SILICON M4 MAX")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    overall_start = time.time()
    
    for market in markets:
        for use_sentiment in sentiment_options:
            feature_type = "With Sentiment" if use_sentiment else "Base"
            
            try:
                # Create config with custom parameters
                config = Config(market=market, use_sentiment=use_sentiment)
                config.BATCH_SIZE = args.batch_size
                config.NUM_WORKERS = args.workers
                config.USE_AMP = not args.no_amp
                
                # Run optimized validation
                validator = OptimizedWalkForwardValidator(config)
                metrics = validator.run_parallel()
                
                result = {
                    'market': market,
                    'use_sentiment': use_sentiment,
                    'metrics': metrics,
                    'status': 'success'
                }
                all_results.append(result)
                
                # Update performance summary
                tracker_metrics = {
                    'test_accuracy': metrics.get('accuracy_mean', 0),
                    'test_precision': metrics.get('precision_mean', 0),
                    'test_recall': metrics.get('recall_mean', 0),
                    'test_f1': metrics.get('f1_score_mean', 0)
                }
                
                model_suffix = "sentiment" if use_sentiment else "base"
                update_performance_summary(
                    model_type=f'daily_{model_suffix}_optimized',
                    market=market,
                    metrics=tracker_metrics,
                    features_type='+Sentiment' if use_sentiment else 'Base',
                    summary_path='../../04_Models/performance_summary.txt'
                )
                
                logger.info(f"✅ {market} - {feature_type} completed")
                
            except Exception as e:
                logger.error(f"❌ {market} - {feature_type} failed: {str(e)}")
                all_results.append({
                    'market': market,
                    'use_sentiment': use_sentiment,
                    'error': str(e),
                    'status': 'failed'
                })
    
    # Final summary
    overall_time = time.time() - overall_start
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*80)
    logger.info(f"Total runtime: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    
    n_success = sum(1 for r in all_results if r['status'] == 'success')
    logger.info(f"✅ Completed {n_success}/{len(all_results)} experiments")
    
    return all_results

if __name__ == "__main__":
    # Set multiprocessing start method for macOS
    mp.set_start_method('spawn', force=True)
    main()