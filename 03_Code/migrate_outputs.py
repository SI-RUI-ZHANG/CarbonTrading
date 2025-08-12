#!/usr/bin/env python3
"""
Migration script to reorganize model outputs to flat directory structure
and clean up legacy files.

This script:
1. Migrates existing model outputs from nested to flat structure
2. Removes duplicate meta_reversal folder
3. Cleans up empty directories
4. Creates a backup before migration
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_backup(base_dir):
    """Create a backup of the entire 04_Models directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{base_dir}_backup_{timestamp}"
    
    if os.path.exists(base_dir):
        logger.info(f"Creating backup at {backup_dir}")
        shutil.copytree(base_dir, backup_dir)
        return backup_dir
    else:
        logger.error(f"Directory {base_dir} does not exist")
        return None

def migrate_daily_weekly_models(base_dir):
    """Migrate daily and weekly models to flat structure"""
    migrations = []
    
    # Migrate daily models
    for market in ['GDEA', 'HBEA']:
        for sentiment in ['base', 'sentiment']:
            old_path = os.path.join(base_dir, 'daily', market, sentiment)
            new_path = os.path.join(base_dir, f'daily_{market}_{sentiment}')
            
            if os.path.exists(old_path):
                logger.info(f"Migrating: {old_path} -> {new_path}")
                shutil.move(old_path, new_path)
                migrations.append((old_path, new_path))
            else:
                logger.warning(f"Path not found: {old_path}")
    
    # Migrate weekly models
    for market in ['GDEA', 'HBEA']:
        for sentiment in ['base', 'sentiment']:
            old_path = os.path.join(base_dir, 'weekly', market, sentiment)
            new_path = os.path.join(base_dir, f'weekly_{market}_{sentiment}')
            
            if os.path.exists(old_path):
                logger.info(f"Migrating: {old_path} -> {new_path}")
                shutil.move(old_path, new_path)
                migrations.append((old_path, new_path))
            else:
                logger.warning(f"Path not found: {old_path}")
    
    return migrations

def migrate_meta_models(base_dir):
    """Migrate meta models to flat structure"""
    migrations = []
    
    # Migrate from meta/ directory
    for freq in ['daily', 'weekly']:
        for market in ['GDEA', 'HBEA']:
            old_path = os.path.join(base_dir, 'meta', freq, market)
            new_path = os.path.join(base_dir, f'meta_{freq}_{market}')
            
            if os.path.exists(old_path):
                logger.info(f"Migrating: {old_path} -> {new_path}")
                shutil.move(old_path, new_path)
                migrations.append((old_path, new_path))
            else:
                logger.warning(f"Path not found: {old_path}")
    
    # Also check meta_reversal directory (duplicate)
    for freq in ['daily', 'weekly']:
        for market in ['GDEA', 'HBEA']:
            old_path = os.path.join(base_dir, 'meta_reversal', freq, market)
            new_path = os.path.join(base_dir, f'meta_{freq}_{market}_reversal')
            
            if os.path.exists(old_path):
                logger.info(f"Found duplicate in meta_reversal: {old_path}")
                # Check if target already exists
                if os.path.exists(new_path.replace('_reversal', '')):
                    logger.warning(f"Target already exists, keeping as: {new_path}")
                    shutil.move(old_path, new_path)
                else:
                    shutil.move(old_path, new_path.replace('_reversal', ''))
                migrations.append((old_path, new_path))
    
    return migrations

def clean_empty_directories(base_dir):
    """Remove empty directories after migration"""
    removed = []
    
    # List of directories that should be removed if empty
    dirs_to_check = [
        'daily/GDEA', 'daily/HBEA', 'daily',
        'weekly/GDEA', 'weekly/HBEA', 'weekly',
        'meta/daily', 'meta/weekly', 'meta',
        'meta_reversal/daily', 'meta_reversal/weekly', 'meta_reversal'
    ]
    
    for dir_name in dirs_to_check:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            logger.info(f"Removing empty directory: {dir_path}")
            os.rmdir(dir_path)
            removed.append(dir_path)
    
    return removed

def identify_legacy_files(base_dir):
    """Identify legacy files that can be removed"""
    legacy_patterns = [
        '*_backup_*',  # Old backup directories
        '*.log',        # Old log files
        '.DS_Store',    # macOS metadata files
        '__pycache__',  # Python cache directories
        '*.pyc',        # Compiled Python files
        'Thumbs.db',    # Windows metadata files
    ]
    
    legacy_files = []
    for root, dirs, files in os.walk(base_dir):
        # Check files
        for file in files:
            file_path = os.path.join(root, file)
            for pattern in legacy_patterns:
                if pattern.startswith('*') and file.endswith(pattern[1:]):
                    legacy_files.append(file_path)
                elif pattern.endswith('*') and file.startswith(pattern[:-1]):
                    legacy_files.append(file_path)
                elif pattern == file:
                    legacy_files.append(file_path)
        
        # Check directories
        dirs_to_remove = []
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for pattern in legacy_patterns:
                if pattern == dir_name or (pattern.startswith('*') and dir_name.endswith(pattern[1:])):
                    legacy_files.append(dir_path)
                    dirs_to_remove.append(dir_name)
        
        # Remove from dirs list to prevent os.walk from descending into them
        for dir_name in dirs_to_remove:
            dirs.remove(dir_name)
    
    return legacy_files

def remove_legacy_files(legacy_files):
    """Remove identified legacy files"""
    removed = []
    for file_path in legacy_files:
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.info(f"Removed directory: {file_path}")
            else:
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
            removed.append(file_path)
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
    
    return removed

def verify_migration(base_dir):
    """Verify that migration was successful"""
    expected_dirs = [
        'daily_GDEA_base', 'daily_GDEA_sentiment',
        'daily_HBEA_base', 'daily_HBEA_sentiment',
        'weekly_GDEA_base', 'weekly_GDEA_sentiment',
        'weekly_HBEA_base', 'weekly_HBEA_sentiment',
        'meta_daily_GDEA', 'meta_daily_HBEA',
        'meta_weekly_GDEA', 'meta_weekly_HBEA'
    ]
    
    missing = []
    for dir_name in expected_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            missing.append(dir_name)
    
    if missing:
        logger.warning(f"Missing expected directories: {missing}")
    else:
        logger.info("All expected directories present after migration")
    
    return len(missing) == 0

def main():
    """Main migration function"""
    base_dir = '../04_Models'
    base_dir = os.path.abspath(base_dir)
    
    logger.info("="*80)
    logger.info("STARTING MODEL OUTPUT MIGRATION")
    logger.info("="*80)
    
    # Step 1: Create backup
    backup_dir = create_backup(base_dir)
    if not backup_dir:
        logger.error("Backup failed, aborting migration")
        return
    
    # Step 2: Migrate daily and weekly models
    logger.info("\nMigrating daily and weekly models...")
    daily_weekly_migrations = migrate_daily_weekly_models(base_dir)
    logger.info(f"Migrated {len(daily_weekly_migrations)} daily/weekly model directories")
    
    # Step 3: Migrate meta models
    logger.info("\nMigrating meta models...")
    meta_migrations = migrate_meta_models(base_dir)
    logger.info(f"Migrated {len(meta_migrations)} meta model directories")
    
    # Step 4: Clean empty directories
    logger.info("\nCleaning empty directories...")
    removed_dirs = clean_empty_directories(base_dir)
    logger.info(f"Removed {len(removed_dirs)} empty directories")
    
    # Step 5: Identify and remove legacy files
    logger.info("\nIdentifying legacy files...")
    legacy_files = identify_legacy_files(base_dir)
    if legacy_files:
        logger.info(f"Found {len(legacy_files)} legacy files")
        for file in legacy_files[:10]:  # Show first 10
            logger.info(f"  - {file}")
        if len(legacy_files) > 10:
            logger.info(f"  ... and {len(legacy_files) - 10} more")
        
        response = input("\nRemove legacy files? (y/n): ")
        if response.lower() == 'y':
            removed_files = remove_legacy_files(legacy_files)
            logger.info(f"Removed {len(removed_files)} legacy files")
    else:
        logger.info("No legacy files found")
    
    # Step 6: Verify migration
    logger.info("\nVerifying migration...")
    success = verify_migration(base_dir)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("MIGRATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Backup created at: {backup_dir}")
    logger.info(f"Total directories migrated: {len(daily_weekly_migrations) + len(meta_migrations)}")
    logger.info(f"Empty directories removed: {len(removed_dirs)}")
    logger.info(f"Legacy files removed: {len(legacy_files) if 'removed_files' in locals() else 0}")
    logger.info(f"Migration successful: {success}")
    
    if success:
        logger.info("\nMigration completed successfully!")
        logger.info("You can now delete the backup if everything looks good:")
        logger.info(f"  rm -rf {backup_dir}")
    else:
        logger.warning("\nMigration completed with warnings. Please check the output.")
        logger.warning(f"Backup preserved at: {backup_dir}")

if __name__ == "__main__":
    main()