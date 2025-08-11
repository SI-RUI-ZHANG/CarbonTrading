"""
Meta-Model Pipeline for LSTM Confidence Prediction

This implements a two-stage approach:
1. Base LSTM (no sentiment) predicts price direction
2. Meta-model (sentiment only) predicts if LSTM is correct

The meta-model acts as a confidence filter or reliability predictor.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import json
import os
from typing import Tuple, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetaModelPipeline:
    """
    Pipeline for training and evaluating meta-models that predict
    LSTM prediction reliability using sentiment features
    """
    
    def __init__(self, market: str = 'GDEA'):
        """
        Initialize meta-model pipeline
        
        Args:
            market: 'GDEA' or 'HBEA'
        """
        self.market = market
        self.base_dir = '../..'
        self.output_dir = f'../../04_Models_Meta/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{market}_Meta'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paths
        self.lstm_model_dir = None  # Will be set to best performing daily LSTM
        self.sentiment_data_path = f'{self.base_dir}/02_Data_Processed/10_Sentiment_Final_Merged/{market}_LSTM_with_sentiment.parquet'
        self.base_data_path = f'{self.base_dir}/02_Data_Processed/03_Feature_Engineered/{market}_LSTM_advanced.parquet'
        
        # Data splits (same dates as LSTM)
        self.train_end = '2020-12-31'
        self.val_end = '2022-12-31'
        
        logger.info(f"Initialized MetaModelPipeline for {market}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_base_lstm_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load predictions from the best daily LSTM model (without sentiment)
        
        Returns:
            Tuple of (train_preds, val_preds, test_preds)
        """
        # Find the best performing daily LSTM model
        models_dir = f'{self.base_dir}/04_Models'
        
        # Look for LSTM models for this market
        best_model_dir = None
        best_accuracy = 0
        
        for dir_name in os.listdir(models_dir):
            if self.market in dir_name and 'LSTM' in dir_name:
                metrics_path = os.path.join(models_dir, dir_name, 'metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        if metrics.get('accuracy', 0) > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_model_dir = os.path.join(models_dir, dir_name)
        
        if not best_model_dir:
            raise ValueError(f"No LSTM model found for {self.market}")
        
        self.lstm_model_dir = best_model_dir
        logger.info(f"Using base LSTM model: {os.path.basename(best_model_dir)}")
        logger.info(f"Base LSTM accuracy: {best_accuracy:.4f}")
        
        # Load test predictions and actuals
        test_preds = np.load(os.path.join(best_model_dir, 'test_predictions.npy'))
        test_actuals = np.load(os.path.join(best_model_dir, 'test_actuals.npy'))
        
        # For training and validation, we need to run the model
        # Load the model and generate predictions
        train_preds, val_preds = self._generate_train_val_predictions()
        
        return train_preds, val_preds, test_preds
    
    def _generate_train_val_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for training and validation sets using the base LSTM
        
        Returns:
            Tuple of (train_preds, val_preds)
        """
        # Load the trained model
        model_path = os.path.join(self.lstm_model_dir, 'best_model.pth')
        
        # Load LSTM data (without sentiment)
        X_train = np.load(f'{self.base_dir}/02_Data_Processed/04_LSTM_Ready/{self.market}_X_train.npy')
        y_train = np.load(f'{self.base_dir}/02_Data_Processed/04_LSTM_Ready/{self.market}_y_train.npy')
        X_val = np.load(f'{self.base_dir}/02_Data_Processed/04_LSTM_Ready/{self.market}_X_val.npy')
        y_val = np.load(f'{self.base_dir}/02_Data_Processed/04_LSTM_Ready/{self.market}_y_val.npy')
        
        # Create simple LSTM model for inference
        input_size = X_train.shape[2]
        model = SimpleLSTM(input_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Generate predictions
        with torch.no_grad():
            # Training predictions
            X_train_tensor = torch.FloatTensor(X_train)
            train_output = model(X_train_tensor)
            train_preds = (torch.sigmoid(train_output) >= 0.5).float().numpy().flatten()
            
            # Validation predictions
            X_val_tensor = torch.FloatTensor(X_val)
            val_output = model(X_val_tensor)
            val_preds = (torch.sigmoid(val_output) >= 0.5).float().numpy().flatten()
        
        return train_preds, val_preds
    
    def prepare_meta_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare sentiment features and correctness labels for meta-model
        
        Returns:
            Tuple of (train_df, val_df, test_df) with features and labels
        """
        # Load sentiment data
        sentiment_df = pd.read_parquet(self.sentiment_data_path)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        sentiment_df = sentiment_df.set_index('Date').sort_index()
        
        # Load base data for true labels
        base_df = pd.read_parquet(self.base_data_path)
        base_df['Date'] = pd.to_datetime(base_df['Date'])
        base_df = base_df.set_index('Date').sort_index()
        
        # Define sentiment features
        sentiment_features = [
            'sentiment_supply', 'sentiment_demand', 'sentiment_policy',
            'supply_decayed', 'demand_decayed', 'policy_decayed',
            'market_pressure', 'pressure_magnitude', 'news_shock',
            'pressure_momentum', 'supply_momentum', 'demand_momentum',
            'doc_count', 'max_policy', 'avg_policy'
        ]
        
        # Get available sentiment features
        available_sentiment = [f for f in sentiment_features if f in sentiment_df.columns]
        logger.info(f"Using {len(available_sentiment)} sentiment features")
        
        # Create target (price direction)
        sentiment_df['target'] = (sentiment_df['log_return'] > 0).astype(int)
        
        # Split data chronologically
        train_df = sentiment_df[sentiment_df.index <= self.train_end].copy()
        val_df = sentiment_df[(sentiment_df.index > self.train_end) & 
                             (sentiment_df.index <= self.val_end)].copy()
        test_df = sentiment_df[sentiment_df.index > self.val_end].copy()
        
        # Account for LSTM sequence length (60 days lookback)
        train_df = train_df.iloc[60:]
        val_df = val_df.iloc[60:]
        test_df = test_df.iloc[60:]
        
        # Select sentiment features and target
        train_meta = train_df[available_sentiment + ['target']].copy()
        val_meta = val_df[available_sentiment + ['target']].copy()
        test_meta = test_df[available_sentiment + ['target']].copy()
        
        logger.info(f"Meta-model data shapes - Train: {train_meta.shape}, Val: {val_meta.shape}, Test: {test_meta.shape}")
        
        return train_meta, val_meta, test_meta
    
    def create_correctness_labels(self, predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
        """
        Create labels indicating whether LSTM predictions were correct
        
        Args:
            predictions: LSTM predictions (0 or 1)
            actuals: True labels (0 or 1)
            
        Returns:
            Correctness labels (1 if correct, 0 if incorrect)
        """
        return (predictions == actuals).astype(int)
    
    def train_logistic_meta(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train logistic regression meta-model"""
        logger.info("Training Logistic Regression meta-model...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        metrics = {
            'model_type': 'logistic_regression',
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        # Save model
        joblib.dump(model, os.path.join(self.output_dir, 'logistic_meta.pkl'))
        joblib.dump(scaler, os.path.join(self.output_dir, 'logistic_scaler.pkl'))
        
        return metrics
    
    def train_random_forest_meta(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest meta-model"""
        logger.info("Training Random Forest meta-model...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics = {
            'model_type': 'random_forest',
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba),
            'top_features': feature_importance.head(5).to_dict('records')
        }
        
        # Save model
        joblib.dump(model, os.path.join(self.output_dir, 'rf_meta.pkl'))
        
        return metrics
    
    def train_xgboost_meta(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost meta-model"""
        logger.info("Training XGBoost meta-model...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'model_type': 'xgboost',
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        # Save model
        model.save_model(os.path.join(self.output_dir, 'xgb_meta.json'))
        
        return metrics
    
    def evaluate_combined_system(self, meta_model, lstm_preds: np.ndarray, 
                                actuals: np.ndarray, sentiment_features: np.ndarray,
                                confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Evaluate the combined LSTM + Meta-model system
        
        Args:
            meta_model: Trained meta-model
            lstm_preds: LSTM predictions
            actuals: True labels
            sentiment_features: Sentiment features for meta-model
            confidence_threshold: Only trade when meta-model confidence > threshold
            
        Returns:
            Performance metrics
        """
        # Get meta-model confidence scores
        if hasattr(meta_model, 'predict_proba'):
            confidence_scores = meta_model.predict_proba(sentiment_features)[:, 1]
        else:
            confidence_scores = meta_model.predict(sentiment_features)
        
        # Filter predictions by confidence
        high_confidence_mask = confidence_scores >= confidence_threshold
        
        if high_confidence_mask.sum() == 0:
            return {
                'filtered_trades': 0,
                'total_trades': len(lstm_preds),
                'filter_rate': 1.0,
                'filtered_accuracy': 0.0
            }
        
        filtered_preds = lstm_preds[high_confidence_mask]
        filtered_actuals = actuals[high_confidence_mask]
        
        # Calculate metrics on filtered predictions
        filtered_accuracy = accuracy_score(filtered_actuals, filtered_preds)
        
        # Calculate improvement
        base_accuracy = accuracy_score(actuals, lstm_preds)
        improvement = filtered_accuracy - base_accuracy
        
        return {
            'base_lstm_accuracy': base_accuracy,
            'filtered_accuracy': filtered_accuracy,
            'improvement': improvement,
            'filtered_trades': high_confidence_mask.sum(),
            'total_trades': len(lstm_preds),
            'filter_rate': 1.0 - (high_confidence_mask.sum() / len(lstm_preds)),
            'filtered_precision': precision_score(filtered_actuals, filtered_preds) if len(np.unique(filtered_preds)) > 1 else 0,
            'filtered_recall': recall_score(filtered_actuals, filtered_preds) if len(np.unique(filtered_preds)) > 1 else 0,
            'filtered_f1': f1_score(filtered_actuals, filtered_preds) if len(np.unique(filtered_preds)) > 1 else 0
        }
    
    def run_pipeline(self):
        """Run the complete meta-model pipeline"""
        logger.info("="*60)
        logger.info(f"META-MODEL PIPELINE - {self.market}")
        logger.info("="*60)
        
        # Step 1: Load base LSTM predictions
        logger.info("\nStep 1: Loading base LSTM predictions...")
        try:
            train_lstm_preds, val_lstm_preds, test_lstm_preds = self.load_base_lstm_predictions()
        except Exception as e:
            logger.error(f"Failed to load LSTM predictions: {e}")
            logger.info("Please ensure you have trained a daily LSTM model first")
            return
        
        # Step 2: Prepare meta-model features
        logger.info("\nStep 2: Preparing sentiment features...")
        train_meta, val_meta, test_meta = self.prepare_meta_features()
        
        # Align predictions with meta features (accounting for sequence length)
        min_train = min(len(train_lstm_preds), len(train_meta))
        min_val = min(len(val_lstm_preds), len(val_meta))
        min_test = min(len(test_lstm_preds), len(test_meta))
        
        train_lstm_preds = train_lstm_preds[-min_train:]
        val_lstm_preds = val_lstm_preds[-min_val:]
        test_lstm_preds = test_lstm_preds[-min_test:]
        
        train_meta = train_meta.iloc[-min_train:]
        val_meta = val_meta.iloc[-min_val:]
        test_meta = test_meta.iloc[-min_test:]
        
        # Step 3: Create correctness labels
        logger.info("\nStep 3: Creating correctness labels...")
        train_correctness = self.create_correctness_labels(train_lstm_preds, train_meta['target'].values)
        val_correctness = self.create_correctness_labels(val_lstm_preds, val_meta['target'].values)
        test_correctness = self.create_correctness_labels(test_lstm_preds, test_meta['target'].values)
        
        logger.info(f"Base LSTM accuracy - Train: {train_correctness.mean():.4f}, Val: {val_correctness.mean():.4f}, Test: {test_correctness.mean():.4f}")
        
        # Prepare features (drop target column)
        X_train = train_meta.drop('target', axis=1)
        X_val = val_meta.drop('target', axis=1)
        X_test = test_meta.drop('target', axis=1)
        
        # Step 4: Train meta-models
        logger.info("\nStep 4: Training meta-models...")
        
        all_metrics = {}
        
        # Train different meta-models
        try:
            all_metrics['logistic'] = self.train_logistic_meta(X_train, train_correctness, X_val, val_correctness)
        except Exception as e:
            logger.warning(f"Logistic regression failed: {e}")
        
        try:
            all_metrics['random_forest'] = self.train_random_forest_meta(X_train, train_correctness, X_val, val_correctness)
        except Exception as e:
            logger.warning(f"Random Forest failed: {e}")
        
        try:
            all_metrics['xgboost'] = self.train_xgboost_meta(X_train, train_correctness, X_val, val_correctness)
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
        
        # Step 5: Evaluate combined systems
        logger.info("\nStep 5: Evaluating combined systems...")
        
        best_model = None
        best_val_auc = 0
        
        for model_name, metrics in all_metrics.items():
            logger.info(f"\n{model_name.upper()} Meta-Model:")
            logger.info(f"  Validation AUC: {metrics['val_auc']:.4f}")
            logger.info(f"  Can predict LSTM correctness: {metrics['val_accuracy']:.4f}")
            
            if metrics['val_auc'] > best_val_auc:
                best_val_auc = metrics['val_auc']
                best_model = model_name
        
        logger.info(f"\nBest meta-model: {best_model} (AUC: {best_val_auc:.4f})")
        
        # Load best model and evaluate on test set
        if best_model == 'logistic':
            model = joblib.load(os.path.join(self.output_dir, 'logistic_meta.pkl'))
            scaler = joblib.load(os.path.join(self.output_dir, 'logistic_scaler.pkl'))
            X_test_model = scaler.transform(X_test)
        elif best_model == 'random_forest':
            model = joblib.load(os.path.join(self.output_dir, 'rf_meta.pkl'))
            X_test_model = X_test
        else:  # xgboost
            model = xgb.XGBClassifier()
            model.load_model(os.path.join(self.output_dir, 'xgb_meta.json'))
            X_test_model = X_test
        
        # Evaluate combined system with different thresholds
        logger.info("\nEvaluating combined system with confidence filtering:")
        
        results = []
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            combined_metrics = self.evaluate_combined_system(
                model, test_lstm_preds, test_meta['target'].values,
                X_test_model, confidence_threshold=threshold
            )
            combined_metrics['threshold'] = threshold
            results.append(combined_metrics)
            
            logger.info(f"\nThreshold {threshold}:")
            logger.info(f"  Filtered Accuracy: {combined_metrics['filtered_accuracy']:.4f}")
            logger.info(f"  Improvement: {combined_metrics['improvement']:+.4f}")
            logger.info(f"  Trading {combined_metrics['filtered_trades']}/{combined_metrics['total_trades']} signals")
            logger.info(f"  Filter rate: {combined_metrics['filter_rate']:.2%}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'combined_results.csv'), index=False)
        
        # Save summary
        summary = {
            'market': self.market,
            'base_lstm_dir': os.path.basename(self.lstm_model_dir),
            'best_meta_model': best_model,
            'meta_model_metrics': all_metrics,
            'combined_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✅ Meta-model pipeline completed!")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return summary


class SimpleLSTM(nn.Module):
    """Simple LSTM for loading trained models"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def main():
    """Run meta-model pipeline for both markets"""
    
    for market in ['GDEA', 'HBEA']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {market}")
        logger.info(f"{'='*60}")
        
        pipeline = MetaModelPipeline(market=market)
        results = pipeline.run_pipeline()
        
        if results:
            # Print summary
            best_result = max(results['combined_results'], key=lambda x: x['filtered_accuracy'])
            logger.info(f"\nBest result for {market}:")
            logger.info(f"  Threshold: {best_result['threshold']}")
            logger.info(f"  Base LSTM Accuracy: {best_result['base_lstm_accuracy']:.4f}")
            logger.info(f"  Filtered Accuracy: {best_result['filtered_accuracy']:.4f}")
            logger.info(f"  Improvement: {best_result['improvement']:+.4f}")
            logger.info(f"  Trading {best_result['filtered_trades']}/{best_result['total_trades']} signals")


if __name__ == "__main__":
    main()