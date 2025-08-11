#!/usr/bin/env python3
"""
Engineer advanced sentiment features with decay, aggregation, and momentum.

Pipeline:
1. Load document scores from 07_Document_Scores
2. Apply policy-weighted daily aggregation
3. Calculate exponential decay features
4. Add market pressure and momentum
5. Save to 09_Sentiment_Engineered
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

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_SCORES = BASE_DIR / "02_Data_Processed" / "07_Document_Scores" / "document_scores.parquet"
OUTPUT_DIR = BASE_DIR / "02_Data_Processed" / "09_Sentiment_Engineered"
OUTPUT_DAILY = OUTPUT_DIR / "sentiment_daily_features.parquet"
OUTPUT_STATS = OUTPUT_DIR / "feature_statistics.json"


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


def main():
    """Main execution pipeline."""
    logger.info("=" * 60)
    logger.info("SENTIMENT FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load document scores
        df = load_document_scores(INPUT_SCORES)
        
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
        
        # Save results
        daily.to_parquet(OUTPUT_DAILY, index=False)
        logger.info(f"Saved engineered features to {OUTPUT_DAILY}")
        
        with open(OUTPUT_STATS, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved feature statistics to {OUTPUT_STATS}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total days: {len(daily)}")
        logger.info(f"Date range: {daily['effective_date'].min()} to {daily['effective_date'].max()}")
        logger.info(f"Features created: {len(daily.columns) - 1} columns")
        logger.info("\nFeature columns:")
        for col in daily.columns:
            if col != 'effective_date':
                logger.info(f"  - {col}")
        
        # Sample statistics
        logger.info("\nKey statistics:")
        logger.info(f"  Market pressure: mean={daily['market_pressure'].mean():.2f}, "
                   f"std={daily['market_pressure'].std():.2f}")
        logger.info(f"  News shock: mean={daily['news_shock'].mean():.2f}, "
                   f"max={daily['news_shock'].max():.2f}")
        logger.info(f"  Documents per day: mean={daily['doc_count'].mean():.2f}, "
                   f"max={daily['doc_count'].max()}")
        
        logger.info("\n✅ Sentiment feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()