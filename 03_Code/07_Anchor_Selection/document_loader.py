"""Load and prepare filtered documents for anchor selection."""

import json
import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import config

def load_filtered_documents() -> List[Dict]:
    """Load all filtered documents from JSONL files.
    
    IMPORTANT: Only loads documents published before or on 2020-12-31
    to avoid data leakage in anchor selection for LSTM training.
    
    Returns:
        List of document dictionaries with full content
    """
    all_documents = []
    filtered_count = 0
    
    # CRITICAL: Training data cutoff date to avoid data leakage
    TRAINING_CUTOFF = datetime(2020, 12, 31)
    
    # Define sources and their sections
    sources = {
        'MEE': ['Decrees', 'Notices'],
        'HBETS': ['Center_Dynamics'],
        'GZETS': ['Trading_Announcements', 'Center_Dynamics', 'Provincial_Municipal']
    }
    
    for source, sections in sources.items():
        source_path = Path(config.FILTERED_DOCS_PATH) / source
        
        for section in sections:
            file_path = source_path / f"{section}_filtered.jsonl"
            
            if file_path.exists():
                print(f"Loading {source}/{section}...")
                section_docs = 0
                section_filtered = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        doc = json.loads(line.strip())
                        
                        # Parse publish date and filter out documents after training cutoff
                        publish_date_str = doc.get('publish_date', '')
                        if publish_date_str:
                            try:
                                # Parse date string (format: "2020-01-01" or "2020-01-01T00:00:00")
                                if 'T' in publish_date_str:
                                    publish_date = datetime.fromisoformat(publish_date_str.split('T')[0])
                                else:
                                    publish_date = datetime.strptime(publish_date_str, '%Y-%m-%d')
                                
                                # Skip documents after training cutoff
                                if publish_date > TRAINING_CUTOFF:
                                    section_filtered += 1
                                    filtered_count += 1
                                    continue
                            except (ValueError, TypeError) as e:
                                print(f"    Warning: Could not parse date '{publish_date_str}' for doc {doc.get('doc_id', 'unknown')}: {e}")
                                # Skip documents with unparseable dates to be safe
                                section_filtered += 1
                                filtered_count += 1
                                continue
                        else:
                            # Skip documents without dates
                            section_filtered += 1
                            filtered_count += 1
                            continue
                        
                        # Keep full content, no truncation
                        all_documents.append({
                            'doc_id': doc['doc_id'],
                            'source': source,
                            'section': section,
                            'title': doc.get('title', ''),
                            'content': doc.get('content', ''),
                            'publish_date': doc.get('publish_date', ''),
                            'url': doc.get('url', '')
                        })
                        section_docs += 1
                
                print(f"  Loaded {section_docs} documents (filtered out {section_filtered} after {TRAINING_CUTOFF.date()})")
            else:
                print(f"  Warning: File not found - {file_path}")
    
    print(f"\n📊 Document Loading Summary:")
    print(f"  Total documents loaded: {len(all_documents)}")
    print(f"  Documents filtered out (after {TRAINING_CUTOFF.date()}): {filtered_count}")
    print(f"  ⚠️  Using only pre-2021 documents to avoid data leakage in LSTM training")
    
    return all_documents


def create_batches(documents: List[Dict], batch_size: int = None) -> List[List[Dict]]:
    """Split documents into batches for parallel processing.
    
    Args:
        documents: List of document dictionaries
        batch_size: Size of each batch (default from config)
        
    Returns:
        List of document batches
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    batches = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        batches.append(batch)
    
    print(f"Created {len(batches)} batches of size {batch_size}")
    return batches


def save_batch_result(batch_id: int, results: List[Dict]):
    """Save batch processing results.
    
    Args:
        batch_id: Batch identifier
        results: List of classification results
    """
    os.makedirs(config.BATCH_RESULTS_PATH, exist_ok=True)
    
    file_path = Path(config.BATCH_RESULTS_PATH) / f"batch_{batch_id:03d}.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved batch {batch_id} results to {file_path}")


def load_batch_results() -> List[Dict]:
    """Load all batch results.
    
    Returns:
        List of all batch results
    """
    all_results = []
    batch_dir = Path(config.BATCH_RESULTS_PATH)
    
    if not batch_dir.exists():
        return all_results
    
    for batch_file in sorted(batch_dir.glob("batch_*.json")):
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
            all_results.extend(batch_results)
    
    return all_results


if __name__ == "__main__":
    # Test loading documents
    docs = load_filtered_documents()
    print(f"\nSample document:")
    if docs:
        sample = docs[0]
        print(f"  ID: {sample['doc_id']}")
        print(f"  Source: {sample['source']}/{sample['section']}")
        print(f"  Title: {sample['title'][:50]}...")
        print(f"  Content length: {len(sample['content'])} chars")
    
    # Test batch creation
    batches = create_batches(docs[:10], batch_size=3)
    print(f"\nBatch test: Created {len(batches)} batches from 10 documents")