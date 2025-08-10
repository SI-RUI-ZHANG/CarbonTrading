#!/usr/bin/env python3
"""Test run the pipeline components"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing Document Processing Pipeline")
print("=" * 60)

# First, let's just check if the scripts can be imported
try:
    # Test import cleaning script
    print("1. Testing document cleaning script...")
    from datetime import datetime
    start = datetime.now()
    
    # Import and run the cleaner
    import importlib.util
    spec = importlib.util.spec_from_file_location("clean", "01_clean_documents.py")
    clean_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean_module)
    
    # Run cleaning
    cleaner = clean_module.DocumentCleaner()
    cleaner.run_cleaning()
    cleaner.save_statistics()
    cleaner.generate_report()
    
    print(f"   Cleaning completed in {(datetime.now() - start).total_seconds():.1f} seconds")
    
    print("\n2. Testing carbon filtering script...")
    start = datetime.now()
    
    # Import and run the filter
    spec = importlib.util.spec_from_file_location("filter", "02_carbon_filter.py")
    filter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(filter_module)
    
    # Run filtering
    filter = filter_module.CarbonDocumentFilter()
    filter.run_filtering()
    filter.save_statistics()
    filter.generate_report()
    
    print(f"   Filtering completed in {(datetime.now() - start).total_seconds():.1f} seconds")
    
    print("\n" + "=" * 60)
    print("Pipeline test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)