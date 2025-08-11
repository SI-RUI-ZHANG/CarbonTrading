#!/usr/bin/env python3
"""
Prepare time-series sentiment scores with 1-day lag for LSTM integration.

This script:
1. Loads document scores from scoring pipeline
2. Sorts documents by publish_date
3. Adds 1-day lag to create effective_date (avoiding look-ahead bias)
4. Keeps only essential columns for LSTM
5. Saves lagged sentiment scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_PATH = BASE_DIR / "02_Data_Processed" / "07_Document_Scores" / "document_scores.parquet"
OUTPUT_DIR = BASE_DIR / "02_Data_Processed" / "08_Sentiment_Timeseries"
OUTPUT_PATH = OUTPUT_DIR / "sentiment_scores_daily.parquet"


def load_document_scores(input_path: Path) -> pd.DataFrame:
    """Load document scores from parquet file."""
    logger.info(f"Loading document scores from {input_path}")
    
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} documents")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Convert publish_date to datetime if needed
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    
    return df


def prepare_sentiment_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare sentiment time series with 1-day lag.
    
    Args:
        df: Document scores dataframe
        
    Returns:
        DataFrame with effective_date and sentiment scores
    """
    logger.info("Preparing sentiment time series")
    
    # Sort by publish_date
    df = df.sort_values('publish_date').copy()
    logger.info(f"Date range: {df['publish_date'].min()} to {df['publish_date'].max()}")
    
    # Add 1-day lag to avoid look-ahead bias
    # Documents published today affect trading tomorrow
    df['effective_date'] = df['publish_date'] + timedelta(days=1)
    logger.info("Added 1-day lag to create effective_date")
    
    # Select essential columns
    columns_to_keep = [
        'effective_date',
        'doc_id',
        'score_supply',
        'score_demand',
        'score_policy_strength'
    ]
    
    result = df[columns_to_keep].copy()
    
    # Log statistics
    logger.info("\nScore statistics:")
    for col in ['score_supply', 'score_demand', 'score_policy_strength']:
        mean_val = result[col].mean()
        std_val = result[col].std()
        min_val = result[col].min()
        max_val = result[col].max()
        logger.info(f"  {col}: mean={mean_val:.2f}, std={std_val:.2f}, "
                   f"min={min_val:.2f}, max={max_val:.2f}")
    
    # Check for multiple documents on same effective date
    docs_per_date = result.groupby('effective_date').size()
    max_docs = docs_per_date.max()
    dates_with_multiple = (docs_per_date > 1).sum()
    
    logger.info(f"\nDocuments per effective_date:")
    logger.info(f"  Maximum documents on single date: {max_docs}")
    logger.info(f"  Dates with multiple documents: {dates_with_multiple}")
    
    if dates_with_multiple > 0:
        logger.warning(f"Found {dates_with_multiple} dates with multiple documents. "
                      "Consider aggregating in future versions.")
    
    return result


def save_sentiment_scores(df: pd.DataFrame, output_path: Path):
    """Save sentiment scores to parquet file."""
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} sentiment scores to {output_path}")
    
    # Log file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")


def main():
    """Main execution pipeline."""
    logger.info("=" * 60)
    logger.info("SENTIMENT TIME-SERIES PREPARATION")
    logger.info("=" * 60)
    
    try:
        # Load document scores
        df = load_document_scores(INPUT_PATH)
        
        # Prepare sentiment time series
        sentiment_df = prepare_sentiment_timeseries(df)
        
        # Save results
        save_sentiment_scores(sentiment_df, OUTPUT_PATH)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents processed: {len(sentiment_df)}")
        logger.info(f"Date range: {sentiment_df['effective_date'].min()} to "
                   f"{sentiment_df['effective_date'].max()}")
        logger.info(f"Output saved to: {OUTPUT_PATH}")
        logger.info("\n✅ Sentiment time-series preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()