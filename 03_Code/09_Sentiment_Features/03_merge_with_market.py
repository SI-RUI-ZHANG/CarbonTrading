#!/usr/bin/env python3
"""
Merge sentiment features with market data for final LSTM input.
Now uses market-specific sentiment features.

Combines:
- Market features from 03_Feature_Engineered/GDEA_LSTM_advanced.parquet
- GDEA sentiment from 09_Sentiment_Engineered/sentiment_daily_features_GDEA.parquet (MEE + GZETS)
- HBEA sentiment from 09_Sentiment_Engineered/sentiment_daily_features_HBEA.parquet (MEE + HBETS)

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
SENTIMENT_DIR = BASE_DIR / "02_Data_Processed" / "09_Sentiment_Engineered"
OUTPUT_DIR = BASE_DIR / "02_Data_Processed" / "10_Sentiment_Final_Merged"

# Market-specific sentiment files
GDEA_SENTIMENT = SENTIMENT_DIR / "sentiment_daily_features_GDEA.parquet"
HBEA_SENTIMENT = SENTIMENT_DIR / "sentiment_daily_features_HBEA.parquet"

# Input market files
GDEA_MARKET = MARKET_DIR / "GDEA_LSTM_advanced.parquet"
HBEA_MARKET = MARKET_DIR / "HBEA_LSTM_advanced.parquet"

# Output files
GDEA_OUTPUT = OUTPUT_DIR / "GDEA_LSTM_with_sentiment.parquet"
HBEA_OUTPUT = OUTPUT_DIR / "HBEA_LSTM_with_sentiment.parquet"


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
    
    # Fill NaN with 0 (no forward-fill to avoid artificial persistence)
    # Decay features already provide temporal continuity
    for col in sentiment_cols:
        merged[col] = merged[col].fillna(0)
    
    missing_after_fill = merged[sentiment_cols[0]].isna().sum()
    
    logger.info(f"  Trading days: {len(market_df)}")
    logger.info(f"  Days with sentiment: {len(market_df) - missing_before}")
    logger.info(f"  Days filled with 0: {missing_before}")
    logger.info(f"  Days remaining with NaN: {missing_after_fill}")
    
    # Verify no NaN values remain
    nan_counts = merged.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Warning: NaN values remain in columns: {nan_counts[nan_counts > 0].to_dict()}")
    
    return merged


def save_merged_features(df: pd.DataFrame, output_path: Path, market_name: str):
    """Save merged features to parquet."""
    # check if the output directory exists
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    
    # Set Date as index for LSTM compatibility
    df = df.set_index('Date')
    df.to_parquet(output_path, index=True)
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
    logger.info("REGIONAL SENTIMENT-MARKET FEATURE MERGER")
    logger.info("=" * 60)
    
    try:
        # Process GDEA market with GDEA-specific sentiment
        if GDEA_MARKET.exists() and GDEA_SENTIMENT.exists():
            logger.info("\n" + "-" * 40)
            logger.info("Processing GDEA (MEE + GZETS sentiment)")
            logger.info("-" * 40)
            
            gdea_sentiment = load_sentiment_features(GDEA_SENTIMENT)
            gdea_market = load_market_features(GDEA_MARKET, "GDEA")
            gdea_merged = merge_sentiment_with_market(gdea_market, gdea_sentiment, "GDEA")
            validate_merge(gdea_merged, "GDEA")
            save_merged_features(gdea_merged, GDEA_OUTPUT, "GDEA")
        else:
            if not GDEA_MARKET.exists():
                logger.warning(f"GDEA market file not found: {GDEA_MARKET}")
            if not GDEA_SENTIMENT.exists():
                logger.warning(f"GDEA sentiment file not found: {GDEA_SENTIMENT}")
        
        # Process HBEA market with HBEA-specific sentiment
        if HBEA_MARKET.exists() and HBEA_SENTIMENT.exists():
            logger.info("\n" + "-" * 40)
            logger.info("Processing HBEA (MEE + HBETS sentiment)")
            logger.info("-" * 40)
            
            hbea_sentiment = load_sentiment_features(HBEA_SENTIMENT)
            hbea_market = load_market_features(HBEA_MARKET, "HBEA")
            hbea_merged = merge_sentiment_with_market(hbea_market, hbea_sentiment, "HBEA")
            validate_merge(hbea_merged, "HBEA")
            save_merged_features(hbea_merged, HBEA_OUTPUT, "HBEA")
        else:
            if not HBEA_MARKET.exists():
                logger.warning(f"HBEA market file not found: {HBEA_MARKET}")
            if not HBEA_SENTIMENT.exists():
                logger.warning(f"HBEA sentiment file not found: {HBEA_SENTIMENT}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        if GDEA_OUTPUT.exists():
            gdea_final = pd.read_parquet(GDEA_OUTPUT)
            if GDEA_SENTIMENT.exists():
                gdea_sentiment_df = pd.read_parquet(GDEA_SENTIMENT)
                logger.info(f"GDEA final shape: {gdea_final.shape}")
                logger.info(f"  Market-specific sentiment features: {len([c for c in gdea_sentiment_df.columns if c != 'effective_date'])}")
                logger.info(f"  Sources: MEE + GZETS (Guangdong-specific)")
        
        if HBEA_OUTPUT.exists():
            hbea_final = pd.read_parquet(HBEA_OUTPUT)
            if HBEA_SENTIMENT.exists():
                hbea_sentiment_df = pd.read_parquet(HBEA_SENTIMENT)
                logger.info(f"HBEA final shape: {hbea_final.shape}")
                logger.info(f"  Market-specific sentiment features: {len([c for c in hbea_sentiment_df.columns if c != 'effective_date'])}")
                logger.info(f"  Sources: MEE + HBETS (Hubei-specific)")
        
        logger.info("\n✅ Regional sentiment-market merge completed successfully!")
        logger.info("\nRegional separation achieved:")
        logger.info("  - GDEA uses only MEE + GZETS documents")
        logger.info("  - HBEA uses only MEE + HBETS documents")
        logger.info("\nNext steps:")
        logger.info("1. Update 04_LSTM_Model/data_preparation.py to use new files")
        logger.info("2. Retrain LSTM models with regional sentiment features")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()