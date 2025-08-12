#!/usr/bin/env python3
"""
Standardize output file names across all model types.

Standard output structure for all models:
- config.json: Model configuration
- metrics.json: Aggregated metrics
- walk_forward_metrics.json: Detailed walk-forward results (LSTM models)
- test_predictions.npy: Model predictions
- test_actuals.npy: Actual values  
- test_probabilities.npy: Prediction probabilities
- test_dates.npy: Dates for backtesting alignment (NEW)
- best_model.pth/pkl: Saved model weights
- training_history.png: Training curves
- classification_analysis.png: Performance analysis
- walk_forward_performance.png: Walk-forward analysis (LSTM models)
"""

import os
import shutil
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define standard file mappings
STANDARD_MAPPINGS = {
    # Daily/Weekly LSTM models
    'walk_metrics_all.json': 'walk_detailed.json',  # Rename for consistency
    
    # Meta models - standardize naming
    'error_reversal_model.pkl': 'best_model.pkl',
    'error_reversal_model_metadata.json': 'model_metadata.json',
    'error_reversal_analysis.png': 'classification_analysis.png',
    
    # Keep these as-is (already standard)
    'config.json': 'config.json',
    'metrics.json': 'metrics.json',
    'walk_forward_metrics.json': 'walk_forward_metrics.json',
    'test_predictions.npy': 'test_predictions.npy',
    'test_actuals.npy': 'test_actuals.npy',
    'test_probabilities.npy': 'test_probabilities.npy',
    'test_dates.npy': 'test_dates.npy',
    'best_model.pth': 'best_model.pth',
    'training_history.png': 'training_history.png',
    'classification_analysis.png': 'classification_analysis.png',
    'walk_forward_performance.png': 'walk_forward_performance.png'
}

# Files to remove (intermediate/debug files)
FILES_TO_REMOVE = [
    'lstm_preds_train.npy',
    'lstm_preds_val.npy',
    'X_train.npy',
    'X_val.npy',
    'y_train.npy',
    'y_val.npy',
    'val_error_probs.npy',
    'val_predictions.npy'
]

def standardize_directory(dir_path):
    """Standardize file names in a single directory"""
    renamed_count = 0
    removed_count = 0
    
    if not os.path.exists(dir_path):
        logger.warning(f"Directory not found: {dir_path}")
        return renamed_count, removed_count
    
    logger.info(f"Processing: {os.path.basename(dir_path)}")
    
    # First, remove intermediate files
    for file_name in FILES_TO_REMOVE:
        file_path = os.path.join(dir_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_count += 1
            logger.info(f"  Removed: {file_name}")
    
    # Then, rename files to standard names
    for old_name, new_name in STANDARD_MAPPINGS.items():
        old_path = os.path.join(dir_path, old_name)
        new_path = os.path.join(dir_path, new_name)
        
        if os.path.exists(old_path) and old_name != new_name:
            if os.path.exists(new_path):
                logger.warning(f"  Target exists, skipping: {new_name}")
            else:
                shutil.move(old_path, new_path)
                renamed_count += 1
                logger.info(f"  Renamed: {old_name} -> {new_name}")
    
    return renamed_count, removed_count

def check_required_files(dir_path, model_type):
    """Check if directory has all required standard files"""
    required_files = {
        'all': ['config.json', 'metrics.json'],
        'lstm': ['walk_forward_metrics.json', 'test_predictions.npy', 
                 'test_actuals.npy', 'test_probabilities.npy',
                 'best_model.pth'],
        'meta': ['best_model.pkl', 'test_predictions.npy',
                 'test_actuals.npy', 'model_metadata.json']
    }
    
    missing_files = []
    
    # Check common files
    for file_name in required_files['all']:
        if not os.path.exists(os.path.join(dir_path, file_name)):
            missing_files.append(file_name)
    
    # Check model-specific files
    if 'meta' in model_type:
        check_list = required_files['meta']
    else:
        check_list = required_files['lstm']
    
    for file_name in check_list:
        if not os.path.exists(os.path.join(dir_path, file_name)):
            missing_files.append(file_name)
    
    # Check for critical missing file: test_dates.npy
    if not os.path.exists(os.path.join(dir_path, 'test_dates.npy')):
        missing_files.append('test_dates.npy (CRITICAL - needed for backtesting)')
    
    return missing_files

def main():
    """Main standardization function"""
    base_dir = '../04_Models'
    base_dir = os.path.abspath(base_dir)
    
    logger.info("="*80)
    logger.info("STANDARDIZING MODEL OUTPUT FILE NAMES")
    logger.info("="*80)
    
    # Get all model directories
    model_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and ('_' in item):
            model_dirs.append((item, item_path))
    
    # Sort for consistent ordering
    model_dirs.sort()
    
    total_renamed = 0
    total_removed = 0
    
    # Process each directory
    for dir_name, dir_path in model_dirs:
        renamed, removed = standardize_directory(dir_path)
        total_renamed += renamed
        total_removed += removed
    
    logger.info("\n" + "="*80)
    logger.info("CHECKING FOR MISSING STANDARD FILES")
    logger.info("="*80)
    
    # Check for missing files
    all_missing = {}
    for dir_name, dir_path in model_dirs:
        missing = check_required_files(dir_path, dir_name)
        if missing:
            all_missing[dir_name] = missing
            logger.warning(f"\n{dir_name} missing files:")
            for file_name in missing:
                logger.warning(f"  - {file_name}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("STANDARDIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Directories processed: {len(model_dirs)}")
    logger.info(f"Files renamed: {total_renamed}")
    logger.info(f"Intermediate files removed: {total_removed}")
    
    if all_missing:
        logger.warning(f"\nDirectories with missing files: {len(all_missing)}")
        logger.warning("CRITICAL: test_dates.npy is missing from all models!")
        logger.warning("This file is needed for backtesting alignment.")
        logger.warning("Re-run the models with updated code to generate this file.")
    else:
        logger.info("\nAll directories have standard files!")
    
    # Create a standardization report
    report = {
        'directories_processed': len(model_dirs),
        'files_renamed': total_renamed,
        'files_removed': total_removed,
        'missing_files': all_missing,
        'standard_structure': {
            'config.json': 'Model configuration',
            'metrics.json': 'Aggregated performance metrics',
            'walk_forward_metrics.json': 'Detailed walk-forward results',
            'walk_detailed.json': 'Individual walk metrics (optional)',
            'test_predictions.npy': 'Model predictions',
            'test_actuals.npy': 'Actual values',
            'test_probabilities.npy': 'Prediction probabilities',
            'test_dates.npy': 'Dates for backtesting alignment',
            'best_model.pth/pkl': 'Saved model weights',
            'model_metadata.json': 'Model metadata (meta models)',
            'training_history.png': 'Training curves',
            'classification_analysis.png': 'Performance analysis',
            'walk_forward_performance.png': 'Walk-forward analysis'
        }
    }
    
    report_path = os.path.join(base_dir, 'standardization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nStandardization report saved to: {report_path}")
    
    return total_renamed, total_removed, all_missing

if __name__ == "__main__":
    main()