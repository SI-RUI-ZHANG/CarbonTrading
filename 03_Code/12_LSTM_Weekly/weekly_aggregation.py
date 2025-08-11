"""
Weekly Aggregation Pipeline
Aggregates daily carbon market data to weekly frequency
Handles both sentiment and non-sentiment features based on configuration
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import warnings
from config import config

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# ==================================================================================
# AGGREGATION FUNCTIONS
# ==================================================================================

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted mean, handling edge cases"""
    if len(values) == 0 or weights.sum() == 0:
        return 0.0
    return (values * weights).sum() / weights.sum()

def aggregate_price_features(week_data: pd.DataFrame) -> Dict[str, float]:
    """Aggregate price-related features for a week"""
    result = {}
    
    # Last close (Friday close or last available)
    result['close'] = week_data['close'].iloc[-1]
    
    # VWAP average for the week
    if 'vwap' in week_data.columns:
        result['vwap'] = week_data['vwap'].mean()
    
    return result

def aggregate_volume_features(week_data: pd.DataFrame) -> Dict[str, float]:
    """Aggregate volume-related features for a week"""
    result = {}
    
    # Sum volumes
    result['volume_tons'] = week_data['volume_tons'].sum()
    result['turnover_cny'] = week_data['turnover_cny'].sum()
    
    # Last cumulative value
    if 'cum_turnover_cny' in week_data.columns:
        result['cum_turnover_cny'] = week_data['cum_turnover_cny'].iloc[-1]
    
    return result

def aggregate_gap_features(week_data: pd.DataFrame) -> Dict[str, float]:
    """Aggregate gap-related features for a week"""
    result = {}
    
    # Minimum gap days in the week
    if 'gap_days' in week_data.columns:
        result['gap_days'] = week_data['gap_days'].min()
    
    # Minimum gap age
    if 'gap_days_age' in week_data.columns:
        result['gap_days_age'] = week_data['gap_days_age'].min()
    
    return result

def aggregate_macro_features(week_data: pd.DataFrame) -> Dict[str, float]:
    """Aggregate macroeconomic features (take last value as they're already lagged)"""
    result = {}
    
    # Find all macro columns (contain 'ffill_daily')
    macro_cols = [col for col in week_data.columns if config.MACRO_PATTERN in col]
    
    for col in macro_cols:
        # Take last value of the week (already properly lagged)
        result[col] = week_data[col].iloc[-1]
    
    return result

def aggregate_sentiment_features(week_data: pd.DataFrame) -> Dict[str, float]:
    """Aggregate sentiment features for a week"""
    result = {}
    
    if not config.USE_SENTIMENT:
        return result
    
    # Check if sentiment features exist
    if 'doc_count' not in week_data.columns:
        return result
    
    # Sum document count
    result['doc_count'] = week_data['doc_count'].sum()
    
    # For sentiment scores, use weighted average by doc_count if documents exist
    doc_weights = week_data['doc_count'].fillna(0)
    
    sentiment_scores = ['sentiment_supply', 'sentiment_demand', 'sentiment_policy']
    for score in sentiment_scores:
        if score in week_data.columns:
            if doc_weights.sum() > 0:
                result[score] = weighted_mean(week_data[score], doc_weights)
            else:
                result[score] = 0.0
    
    # Decayed features - take weighted average
    decayed_features = ['supply_decayed', 'demand_decayed', 'policy_decayed']
    for feat in decayed_features:
        if feat in week_data.columns:
            if doc_weights.sum() > 0:
                result[feat] = weighted_mean(week_data[feat], doc_weights)
            else:
                result[feat] = 0.0
    
    # Pressure and momentum features - average
    pressure_features = ['market_pressure', 'pressure_magnitude', 'news_shock',
                        'pressure_momentum', 'supply_momentum', 'demand_momentum']
    for feat in pressure_features:
        if feat in week_data.columns:
            result[feat] = week_data[feat].mean()
    
    # Policy strength - max and average
    if 'max_policy' in week_data.columns:
        result['max_policy'] = week_data['max_policy'].max()
    if 'avg_policy' in week_data.columns:
        result['avg_policy'] = week_data['avg_policy'].mean()
    
    return result

def calculate_weekly_technical_indicators(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators on weekly data"""
    df = weekly_df.copy()
    
    # Weekly returns (this week close vs last week close)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 5-week return
    df['return_5w'] = df['close'].pct_change(5)
    
    # RSI on weekly data (using 14-week period)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # Bollinger Bands width (20-week)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / rolling_mean
    
    # Volume SMA (20-week)
    df['volume_sma_20'] = df['volume_tons'].rolling(window=20).mean()
    
    # Drop intermediate columns
    df = df.drop(['bb_upper', 'bb_lower'], axis=1, errors='ignore')
    
    return df

def calculate_weekly_temporal_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal features for weekly data"""
    df = weekly_df.copy()
    
    # Week of year (1-52)
    df['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding for week of year
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Month features (from week end date)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop intermediate columns
    df = df.drop(['week_of_year', 'month'], axis=1, errors='ignore')
    
    return df

# ==================================================================================
# MAIN AGGREGATION PIPELINE
# ==================================================================================

def aggregate_daily_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to aggregate daily data to weekly frequency
    
    Args:
        daily_df: Daily frequency DataFrame with Date index
        
    Returns:
        Weekly aggregated DataFrame
    """
    logger.info(f"Starting weekly aggregation for {config.MARKET}")
    logger.info(f"Sentiment features: {'ENABLED' if config.USE_SENTIMENT else 'DISABLED'}")
    logger.info(f"Input shape: {daily_df.shape}")
    
    # Ensure datetime index
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        if 'Date' in daily_df.columns:
            daily_df = daily_df.set_index('Date')
        daily_df.index = pd.to_datetime(daily_df.index)
    
    # Group by week (ending on Friday)
    # 'W-FRI' means week ending on Friday
    weekly_groups = daily_df.groupby(pd.Grouper(freq='W-FRI'))
    
    weekly_records = []
    skipped_weeks = 0
    
    for week_end, week_data in weekly_groups:
        # Skip weeks with too few trading days
        if len(week_data) < config.MIN_DAYS_PER_WEEK:
            skipped_weeks += 1
            continue
        
        # Initialize record with week end date
        record = {'Date': week_end}
        
        # Aggregate different feature groups
        record.update(aggregate_price_features(week_data))
        record.update(aggregate_volume_features(week_data))
        record.update(aggregate_gap_features(week_data))
        record.update(aggregate_macro_features(week_data))
        
        # Add sentiment features if enabled
        if config.USE_SENTIMENT:
            record.update(aggregate_sentiment_features(week_data))
        
        weekly_records.append(record)
    
    # Create DataFrame from records
    weekly_df = pd.DataFrame(weekly_records)
    weekly_df = weekly_df.set_index('Date')
    
    logger.info(f"Created {len(weekly_df)} weekly records (skipped {skipped_weeks} weeks with < {config.MIN_DAYS_PER_WEEK} days)")
    
    # Calculate technical indicators on weekly data
    logger.info("Calculating weekly technical indicators...")
    weekly_df = calculate_weekly_technical_indicators(weekly_df)
    
    # Calculate temporal features
    logger.info("Calculating temporal features...")
    weekly_df = calculate_weekly_temporal_features(weekly_df)
    
    # Drop any rows with NaN in critical columns (from indicator calculation)
    before_clean = len(weekly_df)
    weekly_df = weekly_df.dropna()
    after_clean = len(weekly_df)
    
    if before_clean > after_clean:
        logger.info(f"Dropped {before_clean - after_clean} rows with NaN values from indicator calculations")
    
    logger.info(f"Final weekly shape: {weekly_df.shape}")
    
    # Log feature summary
    logger.info("\nFeature Summary:")
    logger.info(f"  Price features: {len([c for c in weekly_df.columns if c in config.PRICE_FEATURES])}")
    logger.info(f"  Volume features: {len([c for c in weekly_df.columns if c in config.VOLUME_FEATURES])}")
    logger.info(f"  Macro features: {len([c for c in weekly_df.columns if config.MACRO_PATTERN in c])}")
    
    if config.USE_SENTIMENT:
        sentiment_cols = [c for c in weekly_df.columns if c in config.SENTIMENT_FEATURES]
        logger.info(f"  Sentiment features: {len(sentiment_cols)}")
    
    logger.info(f"  Total features: {len(weekly_df.columns)}")
    
    return weekly_df

# ==================================================================================
# DATA VALIDATION
# ==================================================================================

def validate_weekly_data(weekly_df: pd.DataFrame) -> bool:
    """Validate the weekly aggregated data"""
    logger.info("\nValidating weekly data...")
    
    is_valid = True
    
    # Check for required columns
    required_base = ['close', 'volume_tons', 'log_return']
    for col in required_base:
        if col not in weekly_df.columns:
            logger.error(f"Missing required column: {col}")
            is_valid = False
    
    # Check sentiment features if enabled
    if config.USE_SENTIMENT:
        required_sentiment = ['doc_count', 'sentiment_supply', 'sentiment_demand']
        for col in required_sentiment:
            if col not in weekly_df.columns:
                logger.warning(f"Missing sentiment column: {col}")
    
    # Check for data integrity
    if weekly_df.isnull().any().any():
        null_cols = weekly_df.columns[weekly_df.isnull().any()].tolist()
        logger.warning(f"Columns with null values: {null_cols}")
    
    # Check date continuity (weeks should be ~7 days apart)
    date_diffs = weekly_df.index.to_series().diff()
    unusual_gaps = date_diffs[date_diffs > pd.Timedelta(days=10)]
    if len(unusual_gaps) > 0:
        logger.warning(f"Found {len(unusual_gaps)} unusual gaps between weeks")
    
    # Summary statistics
    logger.info("\nWeekly Data Summary:")
    logger.info(f"  Date range: {weekly_df.index.min().date()} to {weekly_df.index.max().date()}")
    logger.info(f"  Total weeks: {len(weekly_df)}")
    logger.info(f"  Features: {len(weekly_df.columns)}")
    
    # Target distribution
    if 'log_return' in weekly_df.columns:
        up_weeks = (weekly_df['log_return'] > 0).sum()
        down_weeks = (weekly_df['log_return'] <= 0).sum()
        logger.info(f"  UP weeks: {up_weeks} ({up_weeks/len(weekly_df)*100:.1f}%)")
        logger.info(f"  DOWN weeks: {down_weeks} ({down_weeks/len(weekly_df)*100:.1f}%)")
    
    # Document coverage if using sentiment
    if config.USE_SENTIMENT and 'doc_count' in weekly_df.columns:
        weeks_with_docs = (weekly_df['doc_count'] > 0).sum()
        logger.info(f"  Weeks with documents: {weeks_with_docs} ({weeks_with_docs/len(weekly_df)*100:.1f}%)")
    
    return is_valid

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    """Main execution pipeline"""
    logger.info("="*80)
    logger.info(f"WEEKLY AGGREGATION PIPELINE - {config.MARKET}")
    logger.info("="*80)
    logger.info(f"Configuration: {config}")
    
    try:
        # Load daily data
        logger.info(f"\nLoading daily data from: {config.input_daily_path}")
        daily_df = pd.read_parquet(config.input_daily_path)
        
        # Aggregate to weekly
        logger.info("\nAggregating to weekly frequency...")
        weekly_df = aggregate_daily_to_weekly(daily_df)
        
        # Validate
        is_valid = validate_weekly_data(weekly_df)
        
        if is_valid:
            # Save aggregated data
            logger.info(f"\nSaving weekly data to: {config.aggregated_data_path}")
            weekly_df.to_parquet(config.aggregated_data_path)
            logger.info("✅ Weekly aggregation completed successfully!")
            
            # Print sample
            logger.info("\nSample of weekly data (first 5 rows):")
            logger.info(f"\n{weekly_df.head()}")
            
        else:
            logger.error("❌ Validation failed! Please check the data.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()