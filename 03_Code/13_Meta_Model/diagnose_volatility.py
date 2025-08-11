"""
Diagnose why volatility-gated strategy isn't trading
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from config import MetaConfig
from data_preparation import MetaDataPreparer
import xgboost as xgb

# Configuration
config = MetaConfig()
config.MARKET = 'GDEA'

# Load sentiment data
sentiment_df = pd.read_parquet(config.SENTIMENT_DATA_PATH)
sentiment_df.index = pd.to_datetime(sentiment_df.index)

# Calculate returns and volatility
sentiment_df['return'] = sentiment_df['close'].pct_change()
sentiment_df['volatility'] = sentiment_df['return'].rolling(window=20).std()

# Prepare meta-model data
preparer = MetaDataPreparer(config)
X_train, y_train, X_val, y_val, meta_data = preparer.prepare_meta_training_data()

# Get test data
val_size = len(X_val)
test_dates = meta_data.index[-val_size:]
test_predictions = meta_data['prediction'].values[-val_size:]
test_actuals = meta_data['actual'].values[-val_size:]

# Align with sentiment data
test_df = sentiment_df.loc[test_dates].copy()

print("="*60)
print("DATA DIAGNOSTICS")
print("="*60)

print(f"\nTest set size: {len(test_dates)}")
print(f"Date range: {test_dates[0]} to {test_dates[-1]}")

# Check volatility
test_volatility = test_df['volatility'].values
print(f"\nVolatility statistics:")
print(f"  Non-NaN values: {(~np.isnan(test_volatility)).sum()}")
print(f"  Mean: {np.nanmean(test_volatility):.4f}")
print(f"  Std: {np.nanstd(test_volatility):.4f}")
print(f"  Min: {np.nanmin(test_volatility):.4f}")
print(f"  Max: {np.nanmax(test_volatility):.4f}")

# Calculate percentiles
vol_percentile = stats.rankdata(test_volatility, method='average', nan_policy='omit') / (~np.isnan(test_volatility)).sum()
print(f"\nVolatility percentiles:")
print(f"  Low vol (<20th): {(vol_percentile < 0.2).sum()} samples")
print(f"  Medium vol (20-80th): {((vol_percentile >= 0.2) & (vol_percentile < 0.8)).sum()} samples")
print(f"  High vol (>80th): {(vol_percentile >= 0.8).sum()} samples")

# Check sentiment scores
test_sentiment_supply = test_df['sentiment_supply'].fillna(0).values
test_sentiment_demand = test_df['sentiment_demand'].fillna(0).values
test_sentiment_score = test_sentiment_demand - test_sentiment_supply

print(f"\nSentiment score statistics:")
print(f"  Non-zero values: {(test_sentiment_score != 0).sum()}")
print(f"  Mean: {np.mean(test_sentiment_score):.4f}")
print(f"  Std: {np.std(test_sentiment_score):.4f}")
print(f"  Min: {np.min(test_sentiment_score):.4f}")
print(f"  Max: {np.max(test_sentiment_score):.4f}")

# Train simple meta-model
print("\nTraining meta-model...")
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'learning_rate': 0.1,
    'seed': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)

# Get meta confidence
meta_confidence = model.predict(dval)
print(f"\nMeta-model confidence:")
print(f"  Mean: {np.mean(meta_confidence):.4f}")
print(f"  Std: {np.std(meta_confidence):.4f}")
print(f"  Min: {np.min(meta_confidence):.4f}")
print(f"  Max: {np.max(meta_confidence):.4f}")
print(f"  > 0.6: {(meta_confidence > 0.6).sum()} samples")
print(f"  > 0.5: {(meta_confidence > 0.5).sum()} samples")

# Check trading conditions
low_vol_mask = vol_percentile < 0.2
high_confidence = meta_confidence > 0.6

print(f"\nTrading conditions:")
print(f"  Low volatility: {low_vol_mask.sum()} samples")
print(f"  High confidence: {high_confidence.sum()} samples")
print(f"  Both conditions: {(low_vol_mask & high_confidence).sum()} samples")

# Check with relaxed conditions
low_vol_relaxed = vol_percentile < 0.3
confidence_relaxed = meta_confidence > 0.5

print(f"\nRelaxed conditions:")
print(f"  Low volatility (<30th): {low_vol_relaxed.sum()} samples")
print(f"  Confidence (>0.5): {confidence_relaxed.sum()} samples")
print(f"  Both conditions: {(low_vol_relaxed & confidence_relaxed).sum()} samples")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Volatility distribution
axes[0, 0].hist(test_volatility[~np.isnan(test_volatility)], bins=20, edgecolor='black')
axes[0, 0].axvline(np.nanpercentile(test_volatility, 20), color='r', linestyle='--', label='20th percentile')
axes[0, 0].set_xlabel('Volatility')
axes[0, 0].set_title('Volatility Distribution')
axes[0, 0].legend()

# Sentiment distribution
axes[0, 1].hist(test_sentiment_score, bins=20, edgecolor='black')
axes[0, 1].set_xlabel('Sentiment Score')
axes[0, 1].set_title('Sentiment Score Distribution')

# Meta confidence distribution
axes[1, 0].hist(meta_confidence, bins=20, edgecolor='black')
axes[1, 0].axvline(0.6, color='r', linestyle='--', label='Threshold')
axes[1, 0].set_xlabel('Meta Confidence')
axes[1, 0].set_title('Meta-Model Confidence Distribution')
axes[1, 0].legend()

# Volatility vs Confidence scatter
axes[1, 1].scatter(vol_percentile, meta_confidence, alpha=0.5)
axes[1, 1].axvline(0.2, color='r', linestyle='--', alpha=0.5, label='Vol threshold')
axes[1, 1].axhline(0.6, color='g', linestyle='--', alpha=0.5, label='Conf threshold')
axes[1, 1].set_xlabel('Volatility Percentile')
axes[1, 1].set_ylabel('Meta Confidence')
axes[1, 1].set_title('Trading Conditions')
axes[1, 1].legend()

# Highlight trading region
trading_region = (vol_percentile < 0.2) & (meta_confidence > 0.6)
if trading_region.sum() > 0:
    axes[1, 1].scatter(vol_percentile[trading_region], meta_confidence[trading_region], 
                      color='red', s=100, label=f'Trading ({trading_region.sum()} points)')
    axes[1, 1].legend()

plt.tight_layout()
plt.savefig('../../04_Models_Meta/volatility_diagnostic.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nDiagnostic plot saved to ../../04_Models_Meta/volatility_diagnostic.png")