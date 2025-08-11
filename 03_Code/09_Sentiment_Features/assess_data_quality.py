#!/usr/bin/env python3
"""
Assess data quality of sentiment-enhanced market datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GDEA_PATH = BASE_DIR / "02_Data_Processed" / "03_Feature_Engineered" / "GDEA_LSTM_with_sentiment.parquet"
HBEA_PATH = BASE_DIR / "02_Data_Processed" / "03_Feature_Engineered" / "HBEA_LSTM_with_sentiment.parquet"

def assess_dataset(df: pd.DataFrame, name: str):
    """Comprehensive data quality assessment."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} DATA QUALITY ASSESSMENT")
    logger.info(f"{'='*60}")
    
    # Basic info
    logger.info(f"\n📊 BASIC INFO:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    logger.info(f"\n🔍 MISSING VALUES:")
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) > 0:
        logger.info(f"  Columns with NaN: {len(nan_cols)}/{len(df.columns)}")
        for col in nan_cols.index[:10]:  # Show first 10
            pct = (nan_cols[col] / len(df)) * 100
            logger.info(f"    {col}: {nan_cols[col]} ({pct:.1f}%)")
        if len(nan_cols) > 10:
            logger.info(f"    ... and {len(nan_cols)-10} more columns")
    else:
        logger.info("  ✅ No missing values!")
    
    # Sentiment features analysis
    logger.info(f"\n📈 SENTIMENT FEATURES:")
    sentiment_cols = ['sentiment_supply', 'sentiment_demand', 'market_pressure', 
                     'supply_decayed', 'demand_decayed', 'news_shock']
    
    for col in sentiment_cols:
        if col in df.columns:
            non_zero = (df[col] != 0).sum()
            pct_non_zero = (non_zero / len(df)) * 100
            logger.info(f"  {col}:")
            logger.info(f"    Non-zero: {non_zero}/{len(df)} ({pct_non_zero:.1f}%)")
            logger.info(f"    Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            logger.info(f"    Mean±Std: {df[col].mean():.2f} ± {df[col].std():.2f}")
    
    # Check for data anomalies
    logger.info(f"\n⚠️  ANOMALY CHECKS:")
    
    # Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        logger.info(f"  ❌ Columns with infinite values: {inf_cols}")
    else:
        logger.info(f"  ✅ No infinite values")
    
    # Check for duplicate dates
    dup_dates = df['Date'].duplicated().sum()
    if dup_dates > 0:
        logger.info(f"  ❌ Duplicate dates: {dup_dates}")
    else:
        logger.info(f"  ✅ No duplicate dates")
    
    # Check date continuity (trading days)
    date_diff = df['Date'].diff().dt.days
    gaps = date_diff[date_diff > 10]  # More than 10 days gap
    if len(gaps) > 0:
        logger.info(f"  ⚠️  Large date gaps (>10 days): {len(gaps)} occurrences")
        logger.info(f"     Largest gap: {gaps.max()} days")
    else:
        logger.info(f"  ✅ No unusual date gaps")
    
    # Check for outliers in price
    if 'close' in df.columns:
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df['close'] < Q1 - 3*IQR) | (df['close'] > Q3 + 3*IQR)).sum()
        if outliers > 0:
            logger.info(f"  ⚠️  Price outliers (3×IQR): {outliers}")
        else:
            logger.info(f"  ✅ No extreme price outliers")
    
    # Feature correlations with target
    logger.info(f"\n📊 SENTIMENT-PRICE CORRELATIONS:")
    if 'close' in df.columns:
        for col in ['sentiment_supply', 'sentiment_demand', 'market_pressure', 'news_shock']:
            if col in df.columns:
                # Only use non-NaN values for correlation
                valid_mask = ~(df[col].isna() | df['close'].isna())
                if valid_mask.sum() > 100:  # Need enough data points
                    corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, 'close'])
                    logger.info(f"  {col} vs close: {corr:.4f}")
    
    return nan_cols

def fix_critical_issues(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fix critical data quality issues."""
    logger.info(f"\n🔧 FIXING {name} DATA ISSUES:")
    
    fixed = False
    
    # Fix infinite values
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            logger.info(f"  Replacing {inf_mask.sum()} inf values in {col} with NaN")
            df.loc[inf_mask, col] = np.nan
            fixed = True
    
    # Remove duplicate dates (keep first)
    if df['Date'].duplicated().any():
        before = len(df)
        df = df.drop_duplicates(subset=['Date'], keep='first')
        logger.info(f"  Removed {before - len(df)} duplicate dates")
        fixed = True
    
    # Sort by date
    if not df['Date'].is_monotonic_increasing:
        df = df.sort_values('Date')
        logger.info(f"  Sorted by date")
        fixed = True
    
    if not fixed:
        logger.info(f"  ✅ No critical issues to fix")
    
    return df

def main():
    """Main execution."""
    logger.info("DATA QUALITY ASSESSMENT FOR SENTIMENT-ENHANCED DATASETS")
    logger.info("="*60)
    
    # Load and assess GDEA
    gdea_df = pd.read_parquet(GDEA_PATH)
    gdea_nans = assess_dataset(gdea_df, "GDEA")
    
    # Load and assess HBEA  
    hbea_df = pd.read_parquet(HBEA_PATH)
    hbea_nans = assess_dataset(hbea_df, "HBEA")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    # Check if we need to fix issues
    need_fix = False
    
    # Check for critical issues
    for df, name in [(gdea_df, "GDEA"), (hbea_df, "HBEA")]:
        has_inf = any(np.isinf(df[col]).any() for col in df.select_dtypes(include=[np.number]).columns)
        has_dup = df['Date'].duplicated().any()
        
        if has_inf or has_dup:
            need_fix = True
            logger.info(f"\n❌ {name} needs fixing:")
            if has_inf:
                logger.info(f"  - Has infinite values")
            if has_dup:
                logger.info(f"  - Has duplicate dates")
    
    if need_fix:
        logger.info("\n" + "="*60)
        logger.info("APPLYING FIXES")
        logger.info("="*60)
        
        # Fix GDEA
        gdea_fixed = fix_critical_issues(gdea_df, "GDEA")
        gdea_fixed.to_parquet(GDEA_PATH, index=False)
        logger.info(f"\n✅ Saved fixed GDEA to {GDEA_PATH}")
        
        # Fix HBEA
        hbea_fixed = fix_critical_issues(hbea_df, "HBEA")
        hbea_fixed.to_parquet(HBEA_PATH, index=False)
        logger.info(f"✅ Saved fixed HBEA to {HBEA_PATH}")
    else:
        logger.info("\n✅ No critical issues found - data quality is acceptable!")
    
    # Note about NaN values in macro features
    if len(gdea_nans) > 0 or len(hbea_nans) > 0:
        logger.info("\n📝 NOTE ON NaN VALUES:")
        logger.info("  The NaN values in macroeconomic features are expected:")
        logger.info("  - First 15-22 rows: No historical data for lagged features")
        logger.info("  - First 1-14 rows: No data for technical indicators")
        logger.info("  - These will be handled by the LSTM data preparation pipeline")

if __name__ == "__main__":
    main()