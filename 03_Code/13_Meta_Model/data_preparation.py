"""
Data preparation for Meta-Model
Aligns primary LSTM predictions with sentiment features to create training data
"""

import numpy as np
import pandas as pd
import os
import sys
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetaDataPreparer:
    """Prepares data for meta-model training"""
    
    def __init__(self, config):
        """
        Initialize data preparer
        
        Args:
            config: MetaConfig object
        """
        self.config = config
        self.primary_model = None
        self.sentiment_df = None
        self.base_df = None
        self.meta_features = None
        self.meta_targets = None
        
    def load_primary_model(self):
        """Load the trained primary LSTM model"""
        logger.info(f"Loading primary LSTM model from {self.config.PRIMARY_MODEL_DIR}")
        
        # Check if directory exists
        if not os.path.exists(self.config.PRIMARY_MODEL_DIR):
            logger.warning(f"Primary model directory not found: {self.config.PRIMARY_MODEL_DIR}")
            # Try to find the latest model
            import glob
            pattern = os.path.join(self.config.PRIMARY_MODEL_BASE, self.config.PRIMARY_MODEL_PATTERN)
            models = glob.glob(pattern)
            if models:
                self.config.PRIMARY_MODEL_DIR = sorted(models)[-1]
                logger.info(f"Using model: {self.config.PRIMARY_MODEL_DIR}")
            else:
                raise FileNotFoundError(f"No primary model found matching {pattern}")
        
        # Load test predictions and actuals
        test_pred_path = os.path.join(self.config.PRIMARY_MODEL_DIR, 'test_predictions.npy')
        test_actual_path = os.path.join(self.config.PRIMARY_MODEL_DIR, 'test_actuals.npy')
        
        self.test_predictions = np.load(test_pred_path)
        self.test_actuals = np.load(test_actual_path)
        
        logger.info(f"Loaded test predictions: {self.test_predictions.shape}")
        logger.info(f"Test accuracy: {(self.test_predictions == self.test_actuals).mean():.3f}")
        
    def load_sentiment_data(self):
        """Load sentiment features data"""
        logger.info(f"Loading sentiment data from {self.config.SENTIMENT_DATA_PATH}")
        self.sentiment_df = pd.read_parquet(self.config.SENTIMENT_DATA_PATH)
        logger.info(f"Sentiment data shape: {self.sentiment_df.shape}")
        
        # Extract only sentiment features
        sentiment_cols = self.config.SENTIMENT_FEATURES
        available_cols = [col for col in sentiment_cols if col in self.sentiment_df.columns]
        
        if len(available_cols) < len(sentiment_cols):
            missing = set(sentiment_cols) - set(available_cols)
            logger.warning(f"Missing sentiment columns: {missing}")
        
        self.sentiment_features_df = self.sentiment_df[available_cols].copy()
        logger.info(f"Using {len(available_cols)} sentiment features")
        
    def load_base_data(self):
        """Load base LSTM data to get targets"""
        logger.info(f"Loading base data from {self.config.BASE_DATA_PATH}")
        self.base_df = pd.read_parquet(self.config.BASE_DATA_PATH)
        
        # Calculate returns for targets
        self.base_df['log_return'] = np.log(self.base_df['close'] / self.base_df['close'].shift(1))
        self.base_df['target'] = (self.base_df['log_return'].shift(-1) > 0).astype(int)
        
        logger.info(f"Base data shape: {self.base_df.shape}")
        
    def generate_lstm_predictions_for_training(self):
        """
        Generate LSTM predictions for the entire dataset
        We need predictions on train/val data to train the meta-model
        """
        logger.info("Generating LSTM predictions for full dataset...")
        
        # Load the LSTM model
        import torch
        model_path = os.path.join(self.config.PRIMARY_MODEL_DIR, 'best_model.pth')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.info("Using test predictions only for meta-model training")
            return
        
        # Load model weights and config
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # For now, we'll use just the test predictions
        # In a full implementation, we'd run the model on train/val data too
        logger.info("Using test set predictions for initial meta-model development")
        
    def align_predictions_with_sentiment(self):
        """
        Align LSTM predictions with sentiment features
        Creates the dataset for meta-model training
        """
        logger.info("Aligning predictions with sentiment features...")
        
        # Get test set date range
        # The test set is approximately the last 20% of data
        test_start_idx = int(len(self.base_df) * 0.8)
        test_end_idx = test_start_idx + len(self.test_predictions)
        
        # Handle sequence length offset
        test_start_idx += self.config.SEQUENCE_LENGTH
        test_end_idx = min(test_end_idx + self.config.SEQUENCE_LENGTH, len(self.base_df))
        
        # Get dates for test period
        test_dates = self.base_df.index[test_start_idx:test_end_idx]
        
        if len(test_dates) != len(self.test_predictions):
            # Adjust for mismatch
            min_len = min(len(test_dates), len(self.test_predictions))
            test_dates = test_dates[:min_len]
            self.test_predictions = self.test_predictions[:min_len]
            self.test_actuals = self.test_actuals[:min_len]
        
        # Create DataFrame with predictions and actuals
        predictions_df = pd.DataFrame({
            'prediction': self.test_predictions,
            'actual': self.test_actuals,
            'correct': (self.test_predictions == self.test_actuals).astype(int)
        }, index=test_dates)
        
        # Align with sentiment features
        sentiment_aligned = self.sentiment_features_df.loc[test_dates].copy()
        
        # Combine
        meta_data = pd.concat([sentiment_aligned, predictions_df], axis=1)
        
        # Drop any rows with NaN
        before_drop = len(meta_data)
        meta_data = meta_data.dropna()
        after_drop = len(meta_data)
        
        if before_drop != after_drop:
            logger.warning(f"Dropped {before_drop - after_drop} rows with NaN values")
        
        logger.info(f"Created meta-model dataset with {len(meta_data)} samples")
        logger.info(f"Prediction accuracy in dataset: {meta_data['correct'].mean():.3f}")
        
        return meta_data
    
    def prepare_meta_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for meta-model training
        
        Returns:
            X_train, y_train, X_val, y_val for meta-model
        """
        # Load all data
        self.load_sentiment_data()
        self.load_base_data()
        self.load_primary_model()
        
        # Align predictions with sentiment
        meta_data = self.align_predictions_with_sentiment()
        
        # Split features and target
        feature_cols = self.config.SENTIMENT_FEATURES
        feature_cols = [col for col in feature_cols if col in meta_data.columns]
        
        X = meta_data[feature_cols].values
        y = meta_data['correct'].values
        
        # Add additional features
        additional_features = []
        
        # Add prediction confidence (if we had probabilities)
        # For now, add the prediction itself as a feature
        additional_features.append(meta_data['prediction'].values.reshape(-1, 1))
        
        # Add day of week
        dow = pd.to_datetime(meta_data.index).dayofweek.values.reshape(-1, 1)
        additional_features.append(dow)
        
        # Combine all features
        X = np.hstack([X] + additional_features)
        
        # Split into train/val (80/20 of available data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Meta-model training data:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        logger.info(f"  Train accuracy rate: {y_train.mean():.3f}")
        logger.info(f"  Val accuracy rate: {y_val.mean():.3f}")
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Save scaler
        os.makedirs(self.config.output_dir, exist_ok=True)
        scaler_path = os.path.join(self.config.output_dir, 'meta_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save the full meta_data for analysis
        meta_data_path = os.path.join(self.config.output_dir, 'meta_data.parquet')
        meta_data.to_parquet(meta_data_path)
        logger.info(f"Saved meta data to {meta_data_path}")
        
        return X_train, y_train, X_val, y_val, meta_data
    
    def analyze_sentiment_patterns(self, meta_data: pd.DataFrame):
        """
        Analyze how sentiment relates to prediction accuracy
        
        Args:
            meta_data: DataFrame with sentiment features and prediction results
        """
        logger.info("\n" + "="*60)
        logger.info("SENTIMENT PATTERN ANALYSIS")
        logger.info("="*60)
        
        # Correlation analysis
        sentiment_cols = [col for col in self.config.SENTIMENT_FEATURES if col in meta_data.columns]
        
        correlations = {}
        for col in sentiment_cols:
            corr = meta_data[col].corr(meta_data['correct'])
            correlations[col] = corr
            
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.info("\nCorrelation with prediction accuracy:")
        for feature, corr in sorted_corr:
            logger.info(f"  {feature:25s}: {corr:+.3f}")
        
        # When predictions are correct vs incorrect
        correct_mask = meta_data['correct'] == 1
        incorrect_mask = ~correct_mask
        
        logger.info(f"\nSentiment differences (Correct - Incorrect predictions):")
        for col in sentiment_cols[:5]:  # Top 5 features
            mean_correct = meta_data.loc[correct_mask, col].mean()
            mean_incorrect = meta_data.loc[incorrect_mask, col].mean()
            diff = mean_correct - mean_incorrect
            logger.info(f"  {col:25s}: {diff:+.3f}")
        
        # Document presence analysis
        if 'doc_count' in meta_data.columns:
            with_docs = meta_data['doc_count'] > 0
            accuracy_with_docs = meta_data.loc[with_docs, 'correct'].mean()
            accuracy_without_docs = meta_data.loc[~with_docs, 'correct'].mean()
            
            logger.info(f"\nAccuracy by document presence:")
            logger.info(f"  With documents:    {accuracy_with_docs:.3f} ({with_docs.sum()} samples)")
            logger.info(f"  Without documents: {accuracy_without_docs:.3f} ({(~with_docs).sum()} samples)")
        
        return correlations


def main():
    """Main execution"""
    from config import config
    
    logger.info("="*80)
    logger.info("META-MODEL DATA PREPARATION")
    logger.info("="*80)
    logger.info(f"Configuration: {config}")
    
    # Prepare data
    preparer = MetaDataPreparer(config)
    X_train, y_train, X_val, y_val, meta_data = preparer.prepare_meta_training_data()
    
    # Analyze patterns
    correlations = preparer.analyze_sentiment_patterns(meta_data)
    
    # Save prepared data
    output_dir = config.output_dir
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    
    logger.info(f"\n✅ Data preparation completed!")
    logger.info(f"Output saved to: {output_dir}")
    
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    main()