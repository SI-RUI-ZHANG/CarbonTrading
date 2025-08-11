"""Batch processor for parallel document scoring with checkpoint support."""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd

import config
from api_client import DocumentPositioner
from document_scorer import DocumentScorer

class BatchProcessor:
    """Process documents in batches with parallel execution and checkpointing."""
    
    def __init__(self, batch_size: int = None, max_workers: int = None):
        self.batch_size = batch_size or config.DEFAULT_BATCH_SIZE
        self.max_workers = max_workers or config.MAX_PARALLEL_WORKERS
        
        self.position_scorer = DocumentPositioner()
        self.document_scorer = DocumentScorer()
        
        # Track progress
        self.processed_docs = set()
        self.failed_docs = []
        self.all_scores = []
        self.total_batches_processed = 0  # Track total batches across all runs
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def process_documents(self, documents: List[Dict], resume: bool = True) -> Dict:
        """
        Process all documents with parallel execution and checkpointing.
        
        Args:
            documents: List of documents to process
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        # Filter out already processed documents if resuming
        if resume and self.processed_docs:
            print(f"📌 Resuming from checkpoint: {len(self.processed_docs)} already processed")
            documents = [
                doc for doc in documents 
                if doc.get('doc_id') not in self.processed_docs
            ]
        
        if not documents:
            print("✅ All documents already processed!")
            return self._get_results_summary(time.time() - start_time)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(documents)} DOCUMENTS")
        print(f"{'='*60}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Checkpoint every: {config.SAVE_CHECKPOINT_EVERY} documents")
        
        # Process in batches
        for batch_idx in range(0, len(documents), self.batch_size):
            batch = documents[batch_idx:batch_idx + self.batch_size]
            # Use total_batches_processed to ensure unique batch numbers across runs
            self.total_batches_processed += 1
            batch_num = self.total_batches_processed
            current_batch_in_run = batch_idx // self.batch_size + 1
            total_batches_in_run = (len(documents) + self.batch_size - 1) // self.batch_size
            
            print(f"\n📦 Processing batch {current_batch_in_run}/{total_batches_in_run} "
                  f"(global batch #{batch_num}, {len(batch)} documents)")
            
            # Process batch in parallel
            batch_scores = self._process_batch_parallel(batch, batch_num)
            
            # Add to results
            self.all_scores.extend(batch_scores)
            
            # Update processed set
            for score_data in batch_scores:
                if score_data and 'doc_id' in score_data:
                    self.processed_docs.add(score_data['doc_id'])
            
            # Save checkpoint periodically
            if len(self.processed_docs) % config.SAVE_CHECKPOINT_EVERY == 0:
                self._save_checkpoint()
                print(f"  💾 Checkpoint saved: {len(self.processed_docs)} documents processed")
            
            # Save batch results
            self._save_batch_results(batch_scores, batch_num)
        
        # Final save
        self._save_checkpoint()
        self._save_final_results()
        
        elapsed_time = time.time() - start_time
        return self._get_results_summary(elapsed_time)
    
    def _process_batch_parallel(self, batch: List[Dict], batch_num: int) -> List[Dict]:
        """Process a batch of documents in parallel."""
        batch_scores = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scoring tasks
            futures = {
                executor.submit(self._score_single_document, doc): doc
                for doc in batch
            }
            
            completed = 0
            for future in as_completed(futures):
                doc = futures[future]
                completed += 1
                
                try:
                    score_data = future.result()
                    if score_data:
                        batch_scores.append(score_data)
                        print(f"  ✓ [{completed}/{len(batch)}] {doc.get('doc_id', 'unknown')[:50]}: "
                              f"S={score_data['scores']['supply']:.1f}, "
                              f"D={score_data['scores']['demand']:.1f}, "
                              f"P={score_data['scores']['policy_strength']:.1f}")
                    else:
                        self.failed_docs.append(doc.get('doc_id', 'unknown'))
                        print(f"  ✗ [{completed}/{len(batch)}] Failed: {doc.get('doc_id', 'unknown')[:50]}")
                        
                except Exception as e:
                    self.failed_docs.append(doc.get('doc_id', 'unknown'))
                    print(f"  ✗ [{completed}/{len(batch)}] Error: {str(e)[:100]}")
        
        return batch_scores
    
    def _score_single_document(self, document: Dict) -> Optional[Dict]:
        """Score a single document."""
        try:
            # Get positions from LLM
            positions = self.position_scorer.get_document_positions(document)
            
            if not positions:
                return None
            
            # Process positions into scores
            scores = self.document_scorer.score_document(positions)
            
            # Combine results
            return {
                'doc_id': document.get('doc_id', ''),
                'title': document.get('title', ''),
                'source': document.get('source', ''),
                'publish_date': document.get('publish_date', ''),
                'positions': positions,
                'scores': scores,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    Error scoring {document.get('doc_id', 'unknown')}: {str(e)[:200]}")
            return None
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'processed_docs': list(self.processed_docs),
            'failed_docs': self.failed_docs,
            'num_processed': len(self.processed_docs),
            'num_failed': len(self.failed_docs),
            'total_batches_processed': self.total_batches_processed,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config.CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        if config.CHECKPOINT_PATH.exists():
            try:
                with open(config.CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                self.processed_docs = set(checkpoint_data.get('processed_docs', []))
                self.failed_docs = checkpoint_data.get('failed_docs', [])
                self.total_batches_processed = checkpoint_data.get('total_batches_processed', 0)
                
                # Note: We no longer load parquet scores since _save_final_results 
                # now reads all batch files directly for proper merging
                
                print(f"📂 Loaded checkpoint: {len(self.processed_docs)} documents already processed")
                print(f"   Previous batches: {self.total_batches_processed}")
                
            except Exception as e:
                print(f"⚠️  Error loading checkpoint: {e}")
                print("Starting fresh...")
    
    def _save_batch_results(self, batch_scores: List[Dict], batch_num: int):
        """Save results for a single batch."""
        if not batch_scores:
            return
        
        batch_file = config.BATCH_SCORES_PATH / f"batch_{batch_num:04d}.json"
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_scores, f, ensure_ascii=False, indent=2)
    
    def _save_final_results(self):
        """Save all results to final output files by merging ALL batch files."""
        # Read all batch files instead of just using self.all_scores
        all_scores = []
        
        # Get all batch files
        batch_files = sorted(config.BATCH_SCORES_PATH.glob("batch_*.json"))
        
        if not batch_files:
            print("⚠️  No batch files found to merge")
            return
        
        # Read and combine all batch files
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_scores = json.load(f)
                    all_scores.extend(batch_scores)
            except Exception as e:
                print(f"⚠️  Error reading {batch_file.name}: {e}")
        
        if not all_scores:
            print("⚠️  No scores found in batch files")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_scores)
        
        # Remove duplicates based on doc_id (keep first occurrence)
        original_count = len(df)
        df = df.drop_duplicates(subset=['doc_id'], keep='first')
        if original_count > len(df):
            print(f"  Removed {original_count - len(df)} duplicate documents")
        
        # Extract scores into separate columns
        for dim in ['supply', 'demand', 'policy_strength']:
            df[f'score_{dim}'] = df['scores'].apply(lambda x: x.get(dim, 0))
        
        # Save as parquet
        df.to_parquet(config.FINAL_SCORES_PATH, index=False)
        print(f"💾 Saved {len(df)} unique document scores from {len(batch_files)} batch files to {config.FINAL_SCORES_PATH}")
        
        # Show source distribution
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print("\n📊 Source distribution in final merge:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} documents")
        
        # Calculate and save distributions
        distributions = self._calculate_distributions(df)
        with open(config.DISTRIBUTIONS_PATH, 'w', encoding='utf-8') as f:
            json.dump(distributions, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Saved score distributions to {config.DISTRIBUTIONS_PATH}")
    
    def _calculate_distributions(self, df: pd.DataFrame) -> Dict:
        """Calculate score distributions and statistics."""
        distributions = {}
        
        for dim in ['supply', 'demand', 'policy_strength']:
            col = f'score_{dim}'
            if col in df.columns:
                distributions[dim] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75)),
                    'histogram': {
                        str(k): v for k, v in 
                        df[col].value_counts(bins=10).to_dict().items()
                    }
                }
        
        return distributions
    
    def _get_results_summary(self, elapsed_time: float) -> Dict:
        """Generate summary of processing results."""
        summary = {
            'total_documents': len(self.processed_docs) + len(self.failed_docs),
            'processed_successfully': len(self.processed_docs),
            'failed': len(self.failed_docs),
            'success_rate': len(self.processed_docs) / max(len(self.processed_docs) + len(self.failed_docs), 1),
            'elapsed_time_seconds': elapsed_time,
            'avg_time_per_doc': elapsed_time / max(len(self.processed_docs), 1),
            'api_stats': self.position_scorer.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        with open(config.SUMMARY_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary