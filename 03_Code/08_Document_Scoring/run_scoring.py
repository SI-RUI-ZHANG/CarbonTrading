#!/usr/bin/env python3
"""Main script to run document scoring with spectrum positioning."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import config
from batch_processor import BatchProcessor

def load_filtered_documents():
    """Load all filtered carbon-relevant documents."""
    documents = []
    
    # Load from each source directory (correct structure)
    sources = ['MEE', 'HBETS', 'GZETS']
    
    for source in sources:
        source_path = config.FILTERED_DOCS_PATH / source
        if not source_path.exists():
            print(f"⚠️  Warning: {source_path} not found")
            continue
        
        # Find all JSONL files in subdirectories
        jsonl_files = list(source_path.rglob("*_filtered.jsonl"))
        
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            # Add source if not present
                            if 'source' not in doc:
                                doc['source'] = source
                            documents.append(doc)
            except Exception as e:
                print(f"  Error loading {jsonl_file}: {e}")
    
    print(f"📚 Loaded {len(documents)} filtered documents")
    return documents

def main(test_mode: bool = False, test_size: int = 10, batch_size: int = None, 
         max_workers: int = None, resume: bool = True):
    """
    Run the document scoring pipeline.
    
    Args:
        test_mode: If True, only process a subset of documents
        test_size: Number of documents to process in test mode
        batch_size: Documents per batch
        max_workers: Maximum parallel workers
        resume: Whether to resume from checkpoint
    """
    print("=" * 60)
    print("DOCUMENT SCORING WITH SPECTRUM POSITIONING")
    print("=" * 60)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Model: {config.MODEL}")
    print(f"Output directory: {config.OUTPUT_BASE}")
    
    # Load documents
    print("\n" + "=" * 60)
    print("LOADING DOCUMENTS")
    print("=" * 60)
    
    documents = load_filtered_documents()
    
    if not documents:
        print("❌ No documents found to process!")
        return 1
    
    # Apply test mode if requested
    if test_mode:
        documents = documents[:test_size]
        print(f"📝 TEST MODE: Processing only {len(documents)} documents")
    
    # Initialize processor
    processor = BatchProcessor(
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    # Process documents
    results = processor.process_documents(documents, resume=resume)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully processed: {results['processed_successfully']} documents")
    print(f"❌ Failed: {results['failed']} documents")
    print(f"📊 Success rate: {results['success_rate']*100:.1f}%")
    print(f"⏱️  Total time: {results['elapsed_time_seconds']:.1f} seconds")
    print(f"⚡ Avg time per doc: {results['avg_time_per_doc']:.2f} seconds")
    
    # Print API statistics
    api_stats = results['api_stats']
    print(f"\n📈 API Statistics:")
    print(f"  - Total API calls: {api_stats['total_calls']}")
    print(f"  - Rate limit hits: {api_stats['total_rate_limit_hits']}")
    print(f"  - Total wait time: {api_stats['total_wait_time']:.1f} seconds")
    print(f"  - Current rate: {api_stats['current_rate']} calls/sec")
    
    # Load and display score distributions
    if config.DISTRIBUTIONS_PATH.exists():
        with open(config.DISTRIBUTIONS_PATH, 'r', encoding='utf-8') as f:
            distributions = json.load(f)
        
        print(f"\n📊 Score Distributions:")
        for dim in ['supply', 'demand', 'policy_strength']:
            if dim in distributions:
                dist = distributions[dim]
                print(f"\n  {dim.upper()}:")
                print(f"    Mean: {dist['mean']:.2f}")
                print(f"    Std:  {dist['std']:.2f}")
                print(f"    Range: [{dist['min']:.2f}, {dist['max']:.2f}]")
                print(f"    Q25/Q50/Q75: {dist['q25']:.2f} / {dist['q50']:.2f} / {dist['q75']:.2f}")
    
    print(f"\n✅ Results saved to:")
    print(f"  - Scores: {config.FINAL_SCORES_PATH}")
    print(f"  - Summary: {config.SUMMARY_PATH}")
    print(f"  - Distributions: {config.DISTRIBUTIONS_PATH}")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score documents using spectrum positioning")
    parser.add_argument('--test', action='store_true', help='Run in test mode with subset of documents')
    parser.add_argument('--test-size', type=int, default=10, help='Number of documents for test mode')
    parser.add_argument('--batch-size', type=int, help='Documents per batch (default: 50)')
    parser.add_argument('--max-workers', type=int, help='Maximum parallel workers (default: 30)')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoint')
    
    args = parser.parse_args()
    
    exit_code = main(
        test_mode=args.test,
        test_size=args.test_size,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        resume=not args.no_resume
    )
    
    sys.exit(exit_code)