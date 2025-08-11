#!/usr/bin/env python3
"""
Engineer advanced sentiment features with decay, aggregation, and momentum.
Now with regional separation: GDEA uses MEE+GZETS, HBEA uses MEE+HBETS.

Pipeline:
1. Load document scores from 07_Document_Scores
2. Filter by market-relevant sources
3. Apply policy-weighted daily aggregation
4. Calculate exponential decay features
5. Add market pressure and momentum
6. Save to 09_Sentiment_Engineered (separate files per market)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Market-source mapping
MARKET_SOURCES = {
    'GDEA': ['MEE', 'GZETS'],  # Guangdong uses national + Guangdong-specific sources
    'HBEA': ['MEE', 'HBETS']   # Hubei uses national + Hubei-specific sources
}

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_SCORES = BASE_DIR / "02_Data_Processed" / "07_Document_Scores" / "document_scores.parquet"
OUTPUT_DIR = BASE_DIR / "02_Data_Processed" / "09_Sentiment_Engineered"


def load_document_scores(input_path: Path) -> pd.DataFrame:
    """Load document scores from parquet file."""
    logger.info(f"Loading document scores from {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} documents")
    
    # Convert dates
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['effective_date'] = df['publish_date'] + timedelta(days=1)  # 1-day lag
    
    # Rename score columns for clarity
    df = df.rename(columns={
        'score_supply': 'score_supply',
        'score_demand': 'score_demand',
        'score_policy_strength': 'score_policy_strength'
    })
    
    return df


def filter_by_market(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Filter documents by market-relevant sources."""
    sources = MARKET_SOURCES[market]
    df_filtered = df[df['source'].isin(sources)].copy()
    
    logger.info(f"Filtering for {market} market:")
    logger.info(f"  Sources: {sources}")
    logger.info(f"  Documents: {len(df_filtered)} out of {len(df)}")
    
    # Show source breakdown
    source_counts = df_filtered['source'].value_counts()
    for source, count in source_counts.items():
        logger.info(f"    {source}: {count} documents")
    
    return df_filtered


def aggregate_daily_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Policy-weighted daily aggregation.
    
    For days with multiple documents, weight by policy strength.
    """
    logger.info("Aggregating documents to daily level with policy weighting")
    
    def weighted_aggregate(group):
        """Aggregate multiple documents with policy strength weights."""
        # Add 1 to avoid division by zero and give all docs some weight
        weights = group['score_policy_strength'] + 1
        
        return pd.Series({
            'sentiment_supply': np.average(group['score_supply'], weights=weights),
            'sentiment_demand': np.average(group['score_demand'], weights=weights),
            'max_policy': group['score_policy_strength'].max(),
            'avg_policy': group['score_policy_strength'].mean(),
            'doc_count': len(group)
        })
    
    daily = df.groupby('effective_date').apply(weighted_aggregate).reset_index()
    
    logger.info(f"Aggregated to {len(daily)} daily observations")
    logger.info(f"Date range: {daily['effective_date'].min()} to {daily['effective_date'].max()}")
    
    return daily


def calculate_decay_features(df: pd.DataFrame, daily_dates: pd.DataFrame, half_life: int = 7) -> pd.DataFrame:
    """
    Step 2: Calculate exponential decay for all past documents.
    
    For each date, sum the decayed influence of all prior documents.
    """
    logger.info(f"Calculating decay features with half-life={half_life} days")
    
    decay_features = []
    
    # Sort documents by date for efficiency
    df_sorted = df.sort_values('effective_date')
    
    for current_date in daily_dates['effective_date']:
        # Get all documents up to and including current date
        past_docs = df_sorted[df_sorted['effective_date'] <= current_date].copy()
        
        if len(past_docs) == 0:
            decay_features.append({
                'effective_date': current_date,
                'supply_decayed': 0.0,
                'demand_decayed': 0.0,
                'policy_decayed': 0.0
            })
            continue
        
        # Calculate days elapsed
        days_ago = (current_date - past_docs['effective_date']).dt.days
        
        # Exponential decay: 0.5^(days/half_life)
        decay_factor = 0.5 ** (days_ago / half_life)
        
        # Apply decay and sum
        supply_decayed = (past_docs['score_supply'] * decay_factor).sum()
        demand_decayed = (past_docs['score_demand'] * decay_factor).sum()
        policy_decayed = (past_docs['score_policy_strength'] * decay_factor).sum()
        
        decay_features.append({
            'effective_date': current_date,
            'supply_decayed': supply_decayed,
            'demand_decayed': demand_decayed,
            'policy_decayed': policy_decayed
        })
    
    decay_df = pd.DataFrame(decay_features)
    logger.info("Decay features calculated")
    
    return decay_df


def add_market_pressure(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Supply-demand imbalance weighted by policy strength.
    
    Market pressure = (demand - supply) * (policy_strength / 100)
    """
    logger.info("Calculating market pressure index")
    
    # Basic market pressure: demand minus supply
    daily_df['market_pressure'] = (
        (daily_df['sentiment_demand'] - daily_df['sentiment_supply']) * 
        (daily_df['max_policy'] / 100)
    )
    
    # Absolute pressure magnitude (volatility indicator)
    daily_df['pressure_magnitude'] = (
        (daily_df['sentiment_demand'].abs() + daily_df['sentiment_supply'].abs()) * 
        (daily_df['max_policy'] / 100)
    )
    
    return daily_df


def add_momentum_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: News shock and sentiment momentum.
    """
    logger.info("Adding momentum and shock features")
    
    # News intensity shock (deviation from normal)
    daily_df['doc_count_ma7'] = daily_df['doc_count'].rolling(7, min_periods=1).mean()
    daily_df['news_shock'] = daily_df['doc_count'] / (daily_df['doc_count_ma7'] + 0.1)
    
    # Sentiment momentum (3-day change)
    daily_df['pressure_momentum'] = daily_df['market_pressure'].rolling(3, min_periods=1).mean().diff()
    daily_df['supply_momentum'] = daily_df['sentiment_supply'].rolling(3, min_periods=1).mean().diff()
    daily_df['demand_momentum'] = daily_df['sentiment_demand'].rolling(3, min_periods=1).mean().diff()
    
    # Drop intermediate column
    daily_df = daily_df.drop(columns=['doc_count_ma7'])
    
    return daily_df


def calculate_feature_statistics(df: pd.DataFrame) -> dict:
    """Calculate statistics for normalization."""
    stats = {}
    
    feature_cols = [
        'sentiment_supply', 'sentiment_demand', 'max_policy', 'avg_policy',
        'supply_decayed', 'demand_decayed', 'policy_decayed',
        'market_pressure', 'pressure_magnitude',
        'news_shock', 'pressure_momentum', 'supply_momentum', 'demand_momentum'
    ]
    
    for col in feature_cols:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
    
    return stats


def process_market(df_all: pd.DataFrame, market: str) -> tuple:
    """Process sentiment features for a specific market."""
    logger.info(f"\n{'='*40}")
    logger.info(f"Processing {market} Market")
    logger.info(f"{'='*40}")
    
    # Filter documents for this market
    df = filter_by_market(df_all, market)
    
    if len(df) == 0:
        logger.warning(f"No documents found for {market} market!")
        return None, None
    
    # Step 1: Daily aggregation with policy weighting
    daily = aggregate_daily_weighted(df)
    
    # Step 2: Calculate decay features
    decay_df = calculate_decay_features(df, daily, half_life=7)
    daily = daily.merge(decay_df, on='effective_date', how='left')
    
    # Step 3: Add market pressure
    daily = add_market_pressure(daily)
    
    # Step 4: Add momentum features
    daily = add_momentum_features(daily)
    
    # Sort by date
    daily = daily.sort_values('effective_date').reset_index(drop=True)
    
    # Calculate statistics
    stats = calculate_feature_statistics(daily)
    
    return daily, stats


def main():
    """Main execution pipeline."""
    logger.info("=" * 60)
    logger.info("SENTIMENT FEATURE ENGINEERING WITH REGIONAL SEPARATION")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load all document scores once
        df_all = load_document_scores(INPUT_SCORES)
        
        # Process each market separately
        all_stats = {}
        
        for market in ['GDEA', 'HBEA']:
            # Process market
            daily, stats = process_market(df_all, market)
            
            if daily is None:
                continue
            
            # Save market-specific results
            output_path = OUTPUT_DIR / f"sentiment_daily_features_{market}.parquet"
            daily.to_parquet(output_path, index=False)
            logger.info(f"Saved {market} features to {output_path}")
            
            # Save market-specific statistics
            stats_path = OUTPUT_DIR / f"feature_statistics_{market}.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved {market} statistics to {stats_path}")
            
            # Store for summary
            all_stats[market] = {
                'total_days': len(daily),
                'date_range': f"{daily['effective_date'].min()} to {daily['effective_date'].max()}",
                'doc_count_mean': daily['doc_count'].mean(),
                'doc_count_max': int(daily['doc_count'].max()),
                'market_pressure_mean': daily['market_pressure'].mean(),
                'news_shock_max': daily['news_shock'].max()
            }
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        for market, stats in all_stats.items():
            logger.info(f"\n{market} Market:")
            logger.info(f"  Total days: {stats['total_days']}")
            logger.info(f"  Date range: {stats['date_range']}")
            logger.info(f"  Documents per day: mean={stats['doc_count_mean']:.2f}, max={stats['doc_count_max']}")
            logger.info(f"  Market pressure: mean={stats['market_pressure_mean']:.2f}")
            logger.info(f"  News shock max: {stats['news_shock_max']:.2f}")
        
        logger.info("\n✅ Regional sentiment feature engineering completed successfully!")
        logger.info("\nOutput files created:")
        logger.info("  - sentiment_daily_features_GDEA.parquet (MEE + GZETS)")
        logger.info("  - sentiment_daily_features_HBEA.parquet (MEE + HBETS)")
        logger.info("  - feature_statistics_GDEA.json")
        logger.info("  - feature_statistics_HBEA.json")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()