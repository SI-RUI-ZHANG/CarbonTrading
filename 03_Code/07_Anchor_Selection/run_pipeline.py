#!/usr/bin/env python3
"""Run the MapReduce anchor selection pipeline."""

import argparse
import json
import time
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
from document_loader import load_filtered_documents
from group_processor import GroupProcessor, process_group_wrapper
from merger import AnchorMerger

def main(batch_size: int = None, max_parallel: int = None, test_mode: bool = False, test_size: int = 10):
    """
    Run the anchor selection pipeline using MapReduce pattern.
    
    Args:
        batch_size: Documents per group (default: 20)
        max_parallel: Maximum parallel groups to process (default: 100)
        test_mode: If True, only process a subset of documents
        test_size: Number of documents to process in test mode
    """
    # Set defaults
    if batch_size is None:
        batch_size = config.DEFAULT_BATCH_SIZE
    if max_parallel is None:
        max_parallel = config.MAX_PARALLEL_GROUPS
    
    # Validate batch size
    if batch_size < 16:
        print(f"⚠️  Warning: batch_size={batch_size} is less than 16 anchor slots")
        print("This may result in unfilled slots. Consider increasing batch_size.")
    
    print("=" * 60)
    print("MAPREDUCE ANCHOR SELECTION PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Model: {config.MODEL}")
    print(f"Batch size: {batch_size}")
    print(f"Max parallel groups: {max_parallel}")
    
    # Add rate limit information
    print(f"\n📊 Rate Limit Settings:")
    print(f"  - Target: {config.API_CALLS_PER_SECOND * 60} calls/minute")
    print(f"  - Max parallel groups: {max_parallel}")
    print(f"  - Estimated tokens/call: ~750 (classify), ~1500 (compare)")
    print(f"  - Rate limit buffer: {config.RATE_LIMIT_BUFFER * 100}%")
    print(f"  - Auto-recovery on rate limit: Yes")
    
    # Load documents
    print("\n" + "=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)
    
    start_time = time.time()
    documents = load_filtered_documents()
    
    if test_mode:
        documents = documents[:test_size]
        print(f"📝 TEST MODE: Processing only {len(documents)} documents")
    
    load_time = time.time() - start_time
    print(f"Loading time: {load_time:.2f} seconds")
    
    # Create groups
    groups = []
    for i in range(0, len(documents), batch_size):
        group = documents[i:i + batch_size]
        groups.append(group)
    
    print(f"\nProcessing {len(documents)} documents")
    print(f"Created {len(groups)} groups")
    if groups:
        print(f"Group sizes: {batch_size} (last group: {len(groups[-1])} docs)")
    
    # Phase 1: Process groups in parallel
    print("\n" + "=" * 60)
    print("PHASE 1: GROUP PROCESSING")
    print("=" * 60)
    
    start_time = time.time()
    group_results = []
    
    # Create output directories
    os.makedirs(config.OUTPUT_BASE, exist_ok=True)
    os.makedirs(f"{config.OUTPUT_BASE}/group_results", exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all group processing tasks
        futures = {
            executor.submit(process_group_wrapper, (group, i + 1)): i
            for i, group in enumerate(groups)
        }
        
        completed = 0
        for future in as_completed(futures):
            group_id = futures[future]
            try:
                result = future.result()
                group_results.append(result)
                completed += 1
                
                # Save intermediate result
                group_file = Path(config.OUTPUT_BASE) / "group_results" / f"group_{group_id + 1:03d}.json"
                with open(group_file, 'w', encoding='utf-8') as f:
                    # Convert to serializable format
                    serializable_result = {}
                    for dim in result:
                        serializable_result[dim] = {}
                        for cat in result[dim]:
                            if result[dim][cat]:
                                serializable_result[dim][cat] = {
                                    'doc_id': result[dim][cat]['doc_id'],
                                    'title': result[dim][cat].get('title', '')[:100]
                                }
                            else:
                                serializable_result[dim][cat] = None
                    json.dump(serializable_result, f, ensure_ascii=False, indent=2)
                
                print(f"Completed group {completed}/{len(groups)}")
                
            except Exception as e:
                print(f"Error processing group {group_id + 1}: {e}")
                # Add empty result to maintain count
                group_results.append({})
    
    group_time = time.time() - start_time
    print(f"\nGroup processing time: {group_time:.2f} seconds")
    print(f"Average time per group: {group_time/len(groups):.2f} seconds")
    
    # Phase 2: Merge results
    print("\n" + "=" * 60)
    print("PHASE 2: BINARY TOURNAMENT MERGE")
    print("=" * 60)
    
    start_time = time.time()
    merger = AnchorMerger()
    final_anchors = merger.merge_all_groups(group_results)
    merge_time = time.time() - start_time
    
    print(f"\nMerge time: {merge_time:.2f} seconds")
    
    # Generate and display summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    # Generate both JSON and text summaries
    json_summary = merger.generate_summary(final_anchors)
    text_summary = merger.generate_text_summary(final_anchors)
    print(text_summary)
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save final anchors (with just doc_id and content)
    final_output = {}
    for dimension in config.DIMENSIONS:
        final_output[dimension] = {}
        # Get categories for this dimension
        if dimension in config.DIMENSION_CATEGORIES:
            categories = config.DIMENSION_CATEGORIES[dimension].keys()
        else:
            categories = config.CATEGORIES.keys()
        
        for category in categories:
            if dimension in final_anchors and category in final_anchors[dimension]:
                anchor = final_anchors[dimension][category]
            else:
                anchor = None
            
            if anchor:
                final_output[dimension][category] = {
                    'doc_id': anchor['doc_id'],
                    'content': anchor['content']
                }
            else:
                final_output[dimension][category] = None
    
    with open(config.FINAL_ANCHORS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"Saved anchors to {config.FINAL_ANCHORS_PATH}")
    
    # Save JSON summary with full content
    summary_path = Path(config.OUTPUT_BASE) / "anchor_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")
    
    # Save statistics
    stats = {
        'run_date': datetime.now().isoformat(),
        'test_mode': test_mode,
        'documents_processed': len(documents),
        'batch_size': batch_size,
        'max_parallel_groups': max_parallel,
        'num_groups': len(groups),
        'group_processing_time_seconds': group_time,
        'merge_time_seconds': merge_time,
        'total_time_seconds': load_time + group_time + merge_time,
        'avg_time_per_doc': (group_time + merge_time) / len(documents) if documents else 0,
        'anchor_status': {
            'filled': merger.count_filled_slots(final_anchors),
            'total': sum(len(config.DIMENSION_CATEGORIES.get(d, config.CATEGORIES)) for d in config.DIMENSIONS)
        },
        'total_comparisons': merger.total_comparisons
    }
    
    stats_path = Path(config.OUTPUT_BASE) / "run_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Total runtime: {stats['total_time_seconds']:.2f} seconds")
    
    # Display unfilled slots if any
    filled = stats['anchor_status']['filled']
    total = stats['anchor_status']['total']
    if filled < total:
        print(f"\n⚠️  Warning: {total - filled} anchor slots remain empty")
        print("Empty slots:")
        for dimension in config.DIMENSIONS:
            # Get categories for this dimension
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            
            for category in categories:
                if dimension not in final_anchors or category not in final_anchors[dimension] or final_anchors[dimension][category] is None:
                    print(f"  - {dimension}/{category}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MapReduce anchor selection pipeline")
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Documents per group (default: 20, recommended: >16)')
    parser.add_argument('--max-parallel', type=int, default=100,
                       help='Maximum parallel groups to process (default: 100)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with limited documents')
    parser.add_argument('--test-size', type=int, default=10,
                       help='Number of documents to process in test mode')
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size,
        max_parallel=args.max_parallel,
        test_mode=args.test,
        test_size=args.test_size
    )