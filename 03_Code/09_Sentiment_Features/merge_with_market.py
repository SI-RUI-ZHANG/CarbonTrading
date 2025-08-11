#!/usr/bin/env python3
"""
Merge sentiment features with market data for final LSTM input.

Combines:
- Market features from 03_Feature_Engineered/GDEA_LSTM_advanced.parquet
- Sentiment features from 09_Sentiment_Engineered/sentiment_daily_features.parquet

Output:
- 03_Feature_Engineered/GDEA_LSTM_with_sentiment.parquet
- 03_Feature_Engineered/HBEA_LSTM_with_sentiment.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MARKET_DIR = BASE_DIR / "02_Data_Processed" / "03_Feature_Engineered"
SENTIMENT_PATH = BASE_DIR / "02_Data_Processed" / "09_Sentiment_Engineered" / "sentiment_daily_features.parquet"

# Input files
GDEA_MARKET = MARKET_DIR / "GDEA_LSTM_advanced.parquet"
HBEA_MARKET = MARKET_DIR / "HBEA_LSTM_advanced.parquet"

# Output files
GDEA_OUTPUT = MARKET_DIR / "GDEA_LSTM_with_sentiment.parquet"
HBEA_OUTPUT = MARKET_DIR / "HBEA_LSTM_with_sentiment.parquet"


def load_sentiment_features(sentiment_path: Path) -> pd.DataFrame:
    """Load sentiment features and prepare for merging."""
    logger.info(f"Loading sentiment features from {sentiment_path}")
    
    df = pd.read_parquet(sentiment_path)
    
    # Rename effective_date to Date for merging
    df = df.rename(columns={'effective_date': 'Date'})
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Select key features (exclude intermediate columns)
    feature_cols = [
        'Date',
        'sentiment_supply', 'sentiment_demand',
        'supply_decayed', 'demand_decayed', 'policy_decayed',
        'market_pressure', 'pressure_magnitude',
        'news_shock', 
        'pressure_momentum', 'supply_momentum', 'demand_momentum',
        'doc_count', 'max_policy', 'avg_policy'
    ]
    
    # Keep only existing columns
    cols_to_keep = [col for col in feature_cols if col in df.columns]
    df = df[cols_to_keep]
    
    logger.info(f"Loaded {len(df)} days of sentiment features")
    logger.info(f"Sentiment date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Sentiment features: {[col for col in cols_to_keep if col != 'Date']}")
    
    return df


def load_market_features(market_path: Path, market_name: str) -> pd.DataFrame:
    """Load market features."""
    logger.info(f"Loading {market_name} market features from {market_path}")
    
    df = pd.read_parquet(market_path)
    
    # The market data uses index as dates
    df = df.reset_index()
    
    # Handle different date column names
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'Date'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    elif 'Date' not in df.columns:
        # If there's still no Date column, the first column might be the date
        logger.warning(f"No Date column found. Columns: {df.columns.tolist()[:5]}")
    
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"Loaded {len(df)} days of {market_name} market data")
    logger.info(f"Market date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"Market features: {df.columns.tolist()[:10]}... ({len(df.columns)} total)")
    
    return df


def merge_sentiment_with_market(market_df: pd.DataFrame, sentiment_df: pd.DataFrame, market_name: str) -> pd.DataFrame:
    """
    Merge sentiment features with market data.
    
    Strategy:
    1. Left join on market dates (keep all trading days)
    2. Forward-fill sentiment for non-trading days
    3. Fill remaining NaN with 0 (for dates before sentiment data starts)
    """
    logger.info(f"\nMerging sentiment with {market_name} market data")
    
    # Sort both dataframes by date
    market_df = market_df.sort_values('Date')
    sentiment_df = sentiment_df.sort_values('Date')
    
    # Merge: keep all market dates
    merged = market_df.merge(sentiment_df, on='Date', how='left')
    
    # Log merge statistics
    sentiment_cols = [col for col in sentiment_df.columns if col != 'Date']
    missing_before = merged[sentiment_cols[0]].isna().sum()
    
    # Forward-fill sentiment features (sentiment from previous day carries over)
    for col in sentiment_cols:
        merged[col] = merged[col].fillna(method='ffill')
    
    missing_after_ffill = merged[sentiment_cols[0]].isna().sum()
    
    # Fill remaining NaN with 0 (for dates before sentiment data)
    for col in sentiment_cols:
        merged[col] = merged[col].fillna(0)
    
    logger.info(f"  Trading days: {len(market_df)}")
    logger.info(f"  Days with sentiment: {len(market_df) - missing_before}")
    logger.info(f"  Days filled by forward-fill: {missing_before - missing_after_ffill}")
    logger.info(f"  Days filled with 0: {missing_after_ffill}")
    
    # Verify no NaN values remain
    nan_counts = merged.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Warning: NaN values remain in columns: {nan_counts[nan_counts > 0].to_dict()}")
    
    return merged


def save_merged_features(df: pd.DataFrame, output_path: Path, market_name: str):
    """Save merged features to parquet."""
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {market_name} merged features to {output_path}")
    
    # Log statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Output shape: {df.shape}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")


def validate_merge(merged_df: pd.DataFrame, market_name: str):
    """Validate the merged dataset."""
    logger.info(f"\nValidating {market_name} merged data:")
    
    # Check sentiment feature ranges
    sentiment_cols = ['sentiment_supply', 'sentiment_demand', 'market_pressure']
    for col in sentiment_cols:
        if col in merged_df.columns:
            logger.info(f"  {col}: mean={merged_df[col].mean():.2f}, "
                       f"std={merged_df[col].std():.2f}, "
                       f"non-zero={(merged_df[col] != 0).sum()}")
    
    # Check that original market features are preserved
    market_feature_samples = ['close', 'volume_tons', 'gap_days']
    for col in market_feature_samples:
        if col in merged_df.columns:
            has_nan = merged_df[col].isna().any()
            logger.info(f"  {col} preserved: {'✗ Has NaN' if has_nan else '✓ No NaN'}")


def main():
    """Main execution pipeline."""
    logger.info("=" * 60)
    logger.info("SENTIMENT-MARKET FEATURE MERGER")
    logger.info("=" * 60)
    
    try:
        # Load sentiment features once
        sentiment_df = load_sentiment_features(SENTIMENT_PATH)
        
        # Process GDEA market
        if GDEA_MARKET.exists():
            logger.info("\n" + "-" * 40)
            logger.info("Processing GDEA")
            logger.info("-" * 40)
            
            gdea_market = load_market_features(GDEA_MARKET, "GDEA")
            gdea_merged = merge_sentiment_with_market(gdea_market, sentiment_df, "GDEA")
            validate_merge(gdea_merged, "GDEA")
            save_merged_features(gdea_merged, GDEA_OUTPUT, "GDEA")
        else:
            logger.warning(f"GDEA market file not found: {GDEA_MARKET}")
        
        # Process HBEA market
        if HBEA_MARKET.exists():
            logger.info("\n" + "-" * 40)
            logger.info("Processing HBEA")
            logger.info("-" * 40)
            
            hbea_market = load_market_features(HBEA_MARKET, "HBEA")
            hbea_merged = merge_sentiment_with_market(hbea_market, sentiment_df, "HBEA")
            validate_merge(hbea_merged, "HBEA")
            save_merged_features(hbea_merged, HBEA_OUTPUT, "HBEA")
        else:
            logger.warning(f"HBEA market file not found: {HBEA_MARKET}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        if GDEA_OUTPUT.exists():
            gdea_final = pd.read_parquet(GDEA_OUTPUT)
            logger.info(f"GDEA final shape: {gdea_final.shape}")
            logger.info(f"  New sentiment features: {len([c for c in sentiment_df.columns if c != 'Date'])}")
        
        if HBEA_OUTPUT.exists():
            hbea_final = pd.read_parquet(HBEA_OUTPUT)
            logger.info(f"HBEA final shape: {hbea_final.shape}")
            logger.info(f"  New sentiment features: {len([c for c in sentiment_df.columns if c != 'Date'])}")
        
        logger.info("\n✅ Sentiment-market merge completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update 04_LSTM_Model/data_preparation.py to use new files")
        logger.info("2. Retrain LSTM models with sentiment features")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()