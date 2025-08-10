"""Process groups of documents independently for anchor selection."""

from typing import List, Dict, Optional
from api_client import DocumentClassifier
import config

class GroupProcessor:
    """Process a group of documents independently to find local anchors."""
    
    def __init__(self):
        self.classifier = DocumentClassifier()
    
    def process_group(self, documents: List[Dict], group_id: int) -> Dict:
        """
        Process one group of documents independently.
        
        Args:
            documents: List of documents in this group
            group_id: Identifier for this group
            
        Returns:
            Dictionary of local anchors for this group
        """
        # Initialize local anchors for this group only
        local_anchors = {}
        for dimension in config.DIMENSIONS:
            local_anchors[dimension] = {}
            # Use dimension-specific categories if available
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            for category in categories:
                local_anchors[dimension][category] = None
        
        # Step 1: Classify all documents in the group
        print(f"  Group {group_id}: Classifying {len(documents)} documents...")
        classifications = {}
        for i, doc in enumerate(documents):
            try:
                doc_classifications = self.classifier.classify_document(doc)
                classifications[doc['doc_id']] = doc_classifications
                
                if (i + 1) % 5 == 0:
                    print(f"    Group {group_id}: Classified {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                print(f"    Error classifying document {doc.get('doc_id', 'unknown')}: {e}")
                classifications[doc['doc_id']] = []
        
        # Step 2: Collect candidates per slot
        candidates = {}
        for dimension in config.DIMENSIONS:
            candidates[dimension] = {}
            # Use dimension-specific categories if available
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            for category in categories:
                candidates[dimension][category] = []
        
        for doc in documents:
            doc_id = doc['doc_id']
            if doc_id in classifications:
                for dim, cat in classifications[doc_id]:
                    candidates[dim][cat].append(doc)
        
        # Step 3: Select best for each slot (within this group)
        comparison_count = 0
        for dimension in config.DIMENSIONS:
            # Use dimension-specific categories if available
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            
            for category in categories:
                candidate_list = candidates[dimension][category]
                
                if candidate_list:
                    # Select best from candidates
                    best = self.select_best_from_list(
                        candidate_list, dimension, category
                    )
                    local_anchors[dimension][category] = best
                    comparison_count += len(candidate_list) - 1
        
        print(f"  Group {group_id}: Found anchors for {self.count_filled_slots(local_anchors)} slots")
        print(f"  Group {group_id}: Made {comparison_count} comparisons")
        
        return local_anchors
    
    def select_best_from_list(self, docs: List[Dict], dimension: str, category: str) -> Dict:
        """
        Select the best document from a list through sequential comparison.
        
        Args:
            docs: List of candidate documents
            dimension: Dimension to compare for
            category: Category to compare for
            
        Returns:
            The best document from the list
        """
        if len(docs) == 0:
            return None
        if len(docs) == 1:
            return docs[0]
        
        # Start with first document as current best
        best = docs[0]
        
        # Compare with rest sequentially
        for doc in docs[1:]:
            try:
                winner = self.classifier.compare_documents(best, doc, dimension, category)
                if winner == 'B':
                    best = doc
            except Exception as e:
                print(f"      Error comparing documents: {e}")
                # Keep current best if comparison fails
        
        return best
    
    def count_filled_slots(self, anchors: Dict) -> int:
        """Count how many anchor slots are filled."""
        count = 0
        for dimension in anchors:
            for category in anchors[dimension]:
                if anchors[dimension][category] is not None:
                    count += 1
        return count


def process_group_wrapper(args):
    """Wrapper function for parallel processing."""
    documents, group_id = args
    processor = GroupProcessor()
    return processor.process_group(documents, group_id)