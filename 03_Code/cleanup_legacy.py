#!/usr/bin/env python3
"""
Clean up legacy files and backups after migration and standardization.

This script:
1. Identifies backup directories created during migration
2. Lists legacy file patterns for removal
3. Provides safe cleanup with user confirmation
"""

import os
import shutil
import glob
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_backup_directories(base_path):
    """Find all backup directories created during migration"""
    backup_dirs = []
    pattern = os.path.join(os.path.dirname(base_path), '*_backup_*')
    
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            backup_dirs.append(path)
            # Get creation time
            stat = os.stat(path)
            creation_time = datetime.fromtimestamp(stat.st_mtime)
            logger.info(f"Found backup: {os.path.basename(path)} (created: {creation_time})")
    
    return backup_dirs

def find_legacy_files(base_dir):
    """Find legacy files that can be removed"""
    legacy_patterns = {
        # Old nested directory structures (should be empty after migration)
        'empty_dirs': [],
        
        # Temporary files
        'temp_files': [
            '**/*.pyc',
            '**/__pycache__',
            '**/.DS_Store',
            '**/Thumbs.db',
            '**/*.log',
            '**/*~',
            '**/.ipynb_checkpoints'
        ],
        
        # Old intermediate files (already removed by standardization)
        'intermediate': [
            '**/lstm_preds_*.npy',
            '**/X_train.npy',
            '**/X_val.npy',
            '**/y_train.npy', 
            '**/y_val.npy',
            '**/val_*.npy'
        ],
        
        # Old naming patterns (already renamed)
        'old_names': [
            '**/error_reversal_model*.pkl',
            '**/error_reversal_model*.json',
            '**/error_reversal_*.png',
            '**/walk_metrics_all.json'  # Now walk_detailed.json
        ]
    }
    
    found_files = {
        'empty_dirs': [],
        'temp_files': [],
        'intermediate': [],
        'old_names': []
    }
    
    # Find empty directories
    for root, dirs, files in os.walk(base_dir):
        if not dirs and not files:
            found_files['empty_dirs'].append(root)
    
    # Find files matching patterns
    for category, patterns in legacy_patterns.items():
        if category != 'empty_dirs':
            for pattern in patterns:
                matches = glob.glob(os.path.join(base_dir, pattern), recursive=True)
                found_files[category].extend(matches)
    
    return found_files

def calculate_space_usage(paths):
    """Calculate total space used by files/directories"""
    total_size = 0
    
    for path in paths:
        if os.path.isfile(path):
            total_size += os.path.getsize(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
    
    # Convert to MB
    return total_size / (1024 * 1024)

def remove_items(items, item_type="files"):
    """Remove a list of files or directories"""
    removed_count = 0
    failed_count = 0
    
    for item in items:
        try:
            if os.path.isdir(item):
                if not os.listdir(item):  # Only remove if empty
                    os.rmdir(item)
                    logger.info(f"Removed empty directory: {item}")
                else:
                    shutil.rmtree(item)
                    logger.info(f"Removed directory: {item}")
            else:
                os.remove(item)
                logger.info(f"Removed file: {item}")
            removed_count += 1
        except Exception as e:
            logger.error(f"Failed to remove {item}: {e}")
            failed_count += 1
    
    return removed_count, failed_count

def main():
    """Main cleanup function"""
    base_dir = '../04_Models'
    base_dir = os.path.abspath(base_dir)
    
    logger.info("="*80)
    logger.info("LEGACY FILE CLEANUP")
    logger.info("="*80)
    
    # Step 1: Find backup directories
    logger.info("\n1. CHECKING FOR BACKUP DIRECTORIES...")
    backup_dirs = find_backup_directories(base_dir)
    
    if backup_dirs:
        backup_size = calculate_space_usage(backup_dirs)
        logger.info(f"Found {len(backup_dirs)} backup directories using {backup_size:.2f} MB")
        
        # List backups
        for backup in backup_dirs:
            size = calculate_space_usage([backup])
            logger.info(f"  - {os.path.basename(backup)}: {size:.2f} MB")
    else:
        logger.info("No backup directories found")
    
    # Step 2: Find legacy files
    logger.info("\n2. CHECKING FOR LEGACY FILES...")
    legacy_files = find_legacy_files(base_dir)
    
    # Count and report
    total_legacy = 0
    for category, files in legacy_files.items():
        if files:
            logger.info(f"\n{category.upper().replace('_', ' ')}:")
            logger.info(f"  Found {len(files)} items")
            total_legacy += len(files)
            
            # Show first few examples
            for file in files[:3]:
                logger.info(f"    - {os.path.relpath(file, base_dir)}")
            if len(files) > 3:
                logger.info(f"    ... and {len(files) - 3} more")
    
    if total_legacy == 0:
        logger.info("No legacy files found to clean up")
    
    # Step 3: Calculate total space that can be freed
    logger.info("\n3. SPACE ANALYSIS")
    all_items = backup_dirs.copy()
    for files in legacy_files.values():
        all_items.extend(files)
    
    total_space = calculate_space_usage(all_items)
    logger.info(f"Total space that can be freed: {total_space:.2f} MB")
    
    # Step 4: User confirmation
    if backup_dirs or total_legacy > 0:
        logger.info("\n" + "="*80)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*80)
        logger.info(f"Backup directories to remove: {len(backup_dirs)}")
        logger.info(f"Legacy files to remove: {total_legacy}")
        logger.info(f"Total space to free: {total_space:.2f} MB")
        
        print("\nWARNING: This will permanently delete the above items.")
        response = input("Proceed with cleanup? (y/n): ")
        
        if response.lower() == 'y':
            logger.info("\n4. PERFORMING CLEANUP...")
            
            # Remove backups
            if backup_dirs:
                logger.info("\nRemoving backup directories...")
                removed, failed = remove_items(backup_dirs, "directories")
                logger.info(f"Removed {removed} backup directories, {failed} failed")
            
            # Remove legacy files
            for category, files in legacy_files.items():
                if files:
                    logger.info(f"\nRemoving {category}...")
                    removed, failed = remove_items(files, category)
                    logger.info(f"Removed {removed} items, {failed} failed")
            
            logger.info("\n" + "="*80)
            logger.info("CLEANUP COMPLETED")
            logger.info("="*80)
            
            # Verify final state
            remaining_backups = find_backup_directories(base_dir)
            remaining_legacy = find_legacy_files(base_dir)
            remaining_count = len(remaining_backups) + sum(len(f) for f in remaining_legacy.values())
            
            if remaining_count == 0:
                logger.info("✅ All legacy files and backups cleaned successfully!")
            else:
                logger.warning(f"⚠️ {remaining_count} items could not be removed")
        else:
            logger.info("\nCleanup cancelled by user")
    else:
        logger.info("\n✅ No cleanup needed - everything is already clean!")
    
    # Step 5: Final recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)
    logger.info("1. Re-run models with updated code to generate test_dates.npy")
    logger.info("2. Verify flat directory structure is working correctly")
    logger.info("3. Update any backtesting code to use the new structure")
    logger.info("4. Document the new standardized output format")

if __name__ == "__main__":
    main()