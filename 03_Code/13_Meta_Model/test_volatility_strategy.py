"""
Test volatility-gated sentiment strategy with real data
Compares performance across different ensemble strategies
"""

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import MetaConfig
from data_preparation import MetaDataPreparer
from model import XGBoostMetaModel
import xgboost as xgb

def calculate_rolling_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std()

def load_and_prepare_data(config):
    """Load data and prepare for testing"""
    print("\n" + "="*60)
    print("Loading data for volatility strategy testing...")
    
    # Load sentiment data
    sentiment_df = pd.read_parquet(config.SENTIMENT_DATA_PATH)
    # Date is already the index in the parquet file
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    # Calculate returns and volatility
    sentiment_df['return'] = sentiment_df['close'].pct_change()
    sentiment_df['volatility'] = calculate_rolling_volatility(sentiment_df['return'], window=20)
    sentiment_df['vol_percentile'] = sentiment_df['volatility'].rank(pct=True)
    
    # Load prepared meta-model data
    preparer = MetaDataPreparer(config)
    
    # Run the full preparation pipeline
    X_train, y_train, X_val, y_val, meta_data = preparer.prepare_meta_training_data()
    
    # Get test dates and predictions from meta_data
    # The validation set is the last part of meta_data
    val_size = len(X_val)
    test_dates = meta_data.index[-val_size:]
    test_predictions = meta_data['prediction'].values[-val_size:]
    test_actuals = meta_data['actual'].values[-val_size:]
    
    # Use validation set as test set
    X_test = X_val
    y_test = y_val
    
    # Align volatility and sentiment with test dates
    test_df = sentiment_df.loc[test_dates].copy()
    test_volatility = test_df['volatility'].values
    test_sentiment_supply = test_df['sentiment_supply'].fillna(0).values
    test_sentiment_demand = test_df['sentiment_demand'].fillna(0).values
    
    # Combined sentiment score (demand - supply, as higher demand is bullish)
    test_sentiment_score = test_sentiment_demand - test_sentiment_supply
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': test_dates,
        'test_predictions': test_predictions,
        'test_actuals': test_actuals,
        'test_volatility': test_volatility,
        'test_sentiment_score': test_sentiment_score,
        'test_df': test_df
    }

def train_meta_model(X_train, y_train):
    """Train XGBoost meta-model"""
    print("\nTraining meta-model...")
    
    # Split validation
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_sub = X_train[:-val_size]
    y_train_sub = y_train[:-val_size]
    
    # Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    return model

def apply_strategy(lstm_predictions, meta_confidence, volatility, sentiment_score, strategy):
    """Apply ensemble strategy"""
    batch_size = len(lstm_predictions)
    
    if strategy == 'baseline':
        # Just use LSTM predictions
        final_predictions = lstm_predictions.copy()
        
    elif strategy == 'filtered':
        # Only trade when meta-model is confident
        trade_mask = meta_confidence > 0.6
        final_predictions = lstm_predictions.copy()
        final_predictions[~trade_mask] = -1
        
    elif strategy == 'volatility_gated':
        # Calculate percentiles
        from scipy import stats
        vol_percentile = stats.rankdata(volatility, method='average') / len(volatility)
        sentiment_percentile = stats.rankdata(sentiment_score, method='average') / len(sentiment_score)
        
        # Relaxed gate conditions based on diagnostic findings
        low_vol_mask = vol_percentile < 0.4  # Bottom 40% volatility (more inclusive)
        moderate_confidence = meta_confidence > 0.45  # Around mean confidence
        extreme_negative = sentiment_percentile < 0.2  # Bottom 20% sentiment
        extreme_positive = sentiment_percentile > 0.8  # Top 20% sentiment
        
        # Apply strategy
        final_predictions = np.full(batch_size, -1)  # Default no trade
        
        # Normal trades in low vol + moderate confidence
        normal_trade = low_vol_mask & moderate_confidence & ~extreme_negative & ~extreme_positive
        final_predictions[normal_trade] = lstm_predictions[normal_trade]
        
        # Contrarian trades on extreme sentiment in low vol
        contrarian_negative = low_vol_mask & extreme_negative
        final_predictions[contrarian_negative] = 1 - lstm_predictions[contrarian_negative]
        
        # Follow extreme positive sentiment
        bullish_trade = low_vol_mask & extreme_positive
        final_predictions[bullish_trade] = 1  # Always predict up on extreme positive sentiment
        
    elif strategy == 'selective':
        # Follow LSTM only when meta says it's reliable
        reliable_mask = meta_confidence > 0.5
        final_predictions = np.full(batch_size, -1)
        final_predictions[reliable_mask] = lstm_predictions[reliable_mask]
        
    elif strategy == 'contrarian':
        # When meta strongly disagrees, consider inverting
        low_conf_mask = meta_confidence < 0.3
        final_predictions = lstm_predictions.copy()
        final_predictions[low_conf_mask] = 1 - final_predictions[low_conf_mask]
        
    else:
        final_predictions = lstm_predictions.copy()
    
    return final_predictions

def evaluate_strategy(predictions, actuals, strategy_name):
    """Evaluate strategy performance"""
    # Filter out no-trade signals
    trade_mask = predictions != -1
    traded_predictions = predictions[trade_mask]
    traded_actuals = actuals[trade_mask]
    
    # Calculate metrics
    coverage = np.sum(trade_mask) / len(predictions)
    
    if len(traded_predictions) > 0:
        accuracy = np.mean(traded_predictions == traded_actuals)
        # Calculate returns (simplified: +1 for correct, -1 for incorrect)
        returns = np.where(traded_predictions == traded_actuals, 1, -1)
        total_return = np.sum(returns)
        avg_return = np.mean(returns)
        
        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        accuracy = 0
        total_return = 0
        avg_return = 0
        sharpe = 0
    
    return {
        'strategy': strategy_name,
        'coverage': coverage,
        'trades': np.sum(trade_mask),
        'accuracy': accuracy,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe': sharpe
    }

def main():
    """Main testing function"""
    # Configuration
    config = MetaConfig()
    config.MARKET = 'GDEA'  # Use GDEA which has better sentiment data
    
    # Load and prepare data
    data = load_and_prepare_data(config)
    
    # Train meta-model
    meta_model = train_meta_model(data['X_train'], data['y_train'])
    
    # Get meta-model predictions
    dtest = xgb.DMatrix(data['X_test'])
    meta_confidence = meta_model.predict(dtest)
    
    print(f"\nTest set size: {len(data['test_predictions'])} samples")
    print(f"LSTM baseline accuracy: {np.mean(data['test_predictions'] == data['test_actuals']):.2%}")
    
    # Test different strategies
    strategies = ['baseline', 'filtered', 'selective', 'contrarian', 'volatility_gated']
    results = []
    
    print("\n" + "="*60)
    print("Testing ensemble strategies...")
    print("-"*60)
    
    for strategy in strategies:
        predictions = apply_strategy(
            data['test_predictions'],
            meta_confidence,
            data['test_volatility'],
            data['test_sentiment_score'],
            strategy
        )
        
        metrics = evaluate_strategy(predictions, data['test_actuals'], strategy)
        results.append(metrics)
        
        print(f"\n{strategy.upper()}:")
        print(f"  Coverage: {metrics['coverage']:.1%} ({metrics['trades']} trades)")
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Avg Return: {metrics['avg_return']:.3f}")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Coverage comparison
    axes[0, 0].bar(results_df['strategy'], results_df['coverage'], color='steelblue')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Trading Coverage by Strategy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    axes[0, 1].bar(results_df['strategy'], results_df['accuracy'], color='green')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy by Strategy')
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sharpe ratio comparison
    axes[1, 0].bar(results_df['strategy'], results_df['sharpe'], color='purple')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].set_title('Risk-Adjusted Returns')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Coverage vs Accuracy scatter
    axes[1, 1].scatter(results_df['coverage'], results_df['accuracy'], s=100, alpha=0.6)
    for idx, row in results_df.iterrows():
        axes[1, 1].annotate(row['strategy'], 
                           (row['coverage'], row['accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Coverage')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Coverage vs Accuracy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'volatility_strategy_results.png'), dpi=100, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'), index=False)
    
    # Detailed analysis of volatility-gated strategy
    print("\n" + "="*60)
    print("VOLATILITY-GATED STRATEGY ANALYSIS")
    print("="*60)
    
    # Get volatility-gated predictions
    vol_gated_preds = apply_strategy(
        data['test_predictions'],
        meta_confidence,
        data['test_volatility'],
        data['test_sentiment_score'],
        'volatility_gated'
    )
    
    # Analyze when it trades
    trade_mask = vol_gated_preds != -1
    traded_dates = data['test_dates'][trade_mask]
    
    # Check volatility characteristics
    traded_volatility = data['test_volatility'][trade_mask]
    all_volatility = data['test_volatility']
    
    print(f"\nVolatility characteristics:")
    print(f"  Mean volatility when trading: {np.nanmean(traded_volatility):.4f}")
    print(f"  Mean volatility overall: {np.nanmean(all_volatility):.4f}")
    print(f"  Volatility reduction: {(1 - np.nanmean(traded_volatility)/np.nanmean(all_volatility)):.1%}")
    
    # Monthly distribution of trades
    test_df_trades = data['test_df'].copy()
    test_df_trades['trades'] = trade_mask.astype(int)
    monthly_trades = test_df_trades.groupby(test_df_trades.index.month)['trades'].sum()
    
    print(f"\nMonthly trade distribution:")
    for month, count in monthly_trades.items():
        print(f"  Month {month}: {count} trades")
    
    # Performance during different volatility regimes
    vol_quintiles = pd.qcut(data['test_volatility'], q=5, labels=['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)'])
    
    print(f"\nPerformance by volatility quintile:")
    for q in ['Q1(Low)', 'Q2', 'Q3', 'Q4', 'Q5(High)']:
        q_mask = (vol_quintiles == q) & trade_mask
        if q_mask.sum() > 0:
            q_accuracy = np.mean(vol_gated_preds[q_mask] == data['test_actuals'][q_mask])
            print(f"  {q}: {q_mask.sum()} trades, {q_accuracy:.1%} accuracy")
        else:
            print(f"  {q}: No trades")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find best performing strategy
    best_sharpe_idx = results_df['sharpe'].idxmax()
    best_accuracy_idx = results_df['accuracy'].idxmax()
    
    print(f"\nBest Sharpe Ratio: {results_df.loc[best_sharpe_idx, 'strategy']} ({results_df.loc[best_sharpe_idx, 'sharpe']:.2f})")
    print(f"Best Accuracy: {results_df.loc[best_accuracy_idx, 'strategy']} ({results_df.loc[best_accuracy_idx, 'accuracy']:.1%})")
    
    # Compare volatility-gated to baseline
    baseline_metrics = results_df[results_df['strategy'] == 'baseline'].iloc[0]
    vol_gated_metrics = results_df[results_df['strategy'] == 'volatility_gated'].iloc[0]
    
    print(f"\nVolatility-Gated vs Baseline:")
    print(f"  Coverage: {vol_gated_metrics['coverage']:.1%} vs {baseline_metrics['coverage']:.1%}")
    print(f"  Accuracy: {vol_gated_metrics['accuracy']:.1%} vs {baseline_metrics['accuracy']:.1%}")
    print(f"  Sharpe: {vol_gated_metrics['sharpe']:.2f} vs {baseline_metrics['sharpe']:.2f}")
    
    accuracy_improvement = (vol_gated_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy']
    sharpe_improvement = (vol_gated_metrics['sharpe'] - baseline_metrics['sharpe']) / abs(baseline_metrics['sharpe'] + 1e-10)
    
    print(f"\nImprovements:")
    print(f"  Accuracy improvement: {accuracy_improvement:.1%}")
    print(f"  Sharpe improvement: {sharpe_improvement:.1%}")
    
    return results_df

if __name__ == "__main__":
    results = main()
    print("\nAnalysis complete!")