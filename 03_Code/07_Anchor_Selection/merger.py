"""Merge anchor sets using binary tournament."""

from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_client import DocumentClassifier
import config

class AnchorMerger:
    """Merge multiple anchor sets to find the global best anchors."""
    
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.total_comparisons = 0
    
    def merge_all_groups(self, group_results: List[Dict]) -> Dict:
        """
        Merge all group results to final anchors using binary tournament.
        
        Args:
            group_results: List of anchor sets from all groups
            
        Returns:
            Final merged anchor set
        """
        if not group_results:
            return self._empty_anchors()
        
        if len(group_results) == 1:
            return group_results[0]
        
        print(f"\n=== MERGE PHASE ===")
        print(f"Starting with {len(group_results)} group results")
        
        current_round = group_results
        round_num = 1
        
        while len(current_round) > 1:
            next_size = (len(current_round) + 1) // 2
            print(f"\nMerge Round {round_num}: {len(current_round)} → {next_size}")
            
            next_round = []
            round_comparisons = 0
            
            # Process pairs in parallel
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = []
                
                # Submit merge tasks for pairs
                for i in range(0, len(current_round), 2):
                    if i + 1 < len(current_round):
                        # Merge pair
                        future = executor.submit(
                            self.merge_two_sets,
                            current_round[i],
                            current_round[i + 1],
                            f"{round_num}-{i//2}"
                        )
                        futures.append(future)
                    else:
                        # Odd one advances automatically
                        next_round.append(current_round[i])
                        print(f"  Set {i} advances automatically (odd)")
                
                # Collect merged results
                for j, future in enumerate(as_completed(futures)):
                    try:
                        merged, comparisons = future.result()
                        next_round.append(merged)
                        round_comparisons += comparisons
                        print(f"  Completed merge {j + 1}/{len(futures)} ({comparisons} comparisons)")
                    except Exception as e:
                        print(f"  Error in merge: {e}")
                        # Use first set if merge fails
                        next_round.append(current_round[j * 2])
            
            print(f"Round {round_num} complete: {round_comparisons} comparisons")
            self.total_comparisons += round_comparisons
            
            current_round = next_round
            round_num += 1
        
        print(f"\n=== MERGE COMPLETE ===")
        print(f"Total comparisons in merge phase: {self.total_comparisons}")
        
        return current_round[0]
    
    def merge_two_sets(self, set_a: Dict, set_b: Dict, merge_id: str) -> tuple:
        """
        Merge two anchor sets by comparing slot by slot with adaptive filling.
        
        Args:
            set_a: First anchor set
            set_b: Second anchor set
            merge_id: Identifier for this merge
            
        Returns:
            Tuple of (merged set, comparison count)
        """
        # Define adjacency relationships for adaptive filling
        ADJACENT_CATEGORIES = {
            'supply': {
                'restrict': ['reduce'],      # restrict loser can fill reduce
                'reduce': [],                # reduce has no fallback (middle category)
                'increase': [],              # increase has no fallback (middle category)
                'expand': ['increase']       # expand loser can fill increase
            },
            'demand': {
                'restrict': ['reduce'],      # Same logic as supply
                'reduce': [],                
                'increase': [],              
                'expand': ['increase']       
            }
        }
        
        merged = {}
        comparisons = 0
        losers_pool = []  # Track losers for potential reuse
        
        # First pass: normal merge
        for dimension in config.DIMENSIONS:
            merged[dimension] = {}
            
            # Get appropriate categories for this dimension
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            
            for category in categories:
                anchor_a = set_a.get(dimension, {}).get(category)
                anchor_b = set_b.get(dimension, {}).get(category)
                
                if anchor_a is None:
                    merged[dimension][category] = anchor_b
                elif anchor_b is None:
                    merged[dimension][category] = anchor_a
                else:
                    # Both slots filled, need to compare
                    try:
                        winner = self.classifier.compare_documents(
                            anchor_a, anchor_b, dimension, category
                        )
                        if winner == 'A':
                            merged[dimension][category] = anchor_a
                            # Save loser for potential reuse
                            losers_pool.append((anchor_b, dimension, category))
                        else:
                            merged[dimension][category] = anchor_b
                            # Save loser for potential reuse
                            losers_pool.append((anchor_a, dimension, category))
                        comparisons += 1
                    except Exception as e:
                        print(f"    Error comparing {dimension}/{category}: {e}")
                        # Default to first on error
                        merged[dimension][category] = anchor_a
        
        # Second pass: try to fill empty slots with losers from same direction
        for loser_doc, orig_dim, orig_cat in losers_pool:
            # Only apply adaptive filling for supply/demand dimensions
            if orig_dim in ADJACENT_CATEGORIES:
                adjacent_cats = ADJACENT_CATEGORIES[orig_dim].get(orig_cat, [])
                
                for adj_cat in adjacent_cats:
                    # Check if adjacent slot is empty
                    if merged[orig_dim].get(adj_cat) is None:
                        merged[orig_dim][adj_cat] = loser_doc
                        print(f"      Adaptive fill: {orig_dim}/{orig_cat} loser → {adj_cat}")
                        break  # Only fill one slot per loser
        
        return merged, comparisons
    
    def _empty_anchors(self) -> Dict:
        """Create an empty anchor set."""
        anchors = {}
        for dimension in config.DIMENSIONS:
            anchors[dimension] = {}
            # Use dimension-specific categories if available
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension].keys()
            else:
                categories = config.CATEGORIES.keys()
            for category in categories:
                anchors[dimension][category] = None
        return anchors
    
    def count_filled_slots(self, anchors: Dict) -> int:
        """Count how many anchor slots are filled."""
        count = 0
        for dimension in anchors:
            for category in anchors[dimension]:
                if anchors[dimension][category] is not None:
                    count += 1
        return count
    
    def generate_summary(self, anchors: Dict) -> Dict:
        """Generate a JSON summary of the final anchors with full content."""
        filled = self.count_filled_slots(anchors)
        
        # Calculate total slots based on dimension-specific categories
        total = 0
        for dimension in config.DIMENSIONS:
            if dimension in config.DIMENSION_CATEGORIES:
                total += len(config.DIMENSION_CATEGORIES[dimension])
            else:
                total += len(config.CATEGORIES)
        
        # Build JSON structure
        result = {
            "summary": {
                "total_slots": total,
                "filled": filled,
                "fill_rate": f"{filled/total*100:.1f}%",
                "empty": total - filled,
                "total_comparisons": self.total_comparisons
            },
            "anchors": {}
        }
        
        # Add anchor details with full content
        for dimension in config.DIMENSIONS:
            dim_name = config.DIMENSIONS[dimension]['name']
            result["anchors"][dimension] = {
                "name": dim_name,
                "description": config.DIMENSIONS[dimension]['description'],
                "categories": {}
            }
            
            # Get categories for this dimension
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension]
            else:
                categories = config.CATEGORIES
            
            for category, cat_name in categories.items():
                if dimension in anchors and category in anchors[dimension]:
                    anchor = anchors[dimension][category]
                else:
                    anchor = None
                
                if anchor:
                    result["anchors"][dimension]["categories"][category] = {
                        "name": cat_name,
                        "status": "filled",
                        "document": {
                            "doc_id": anchor.get('doc_id', ''),
                            "title": anchor.get('title', ''),
                            "content": anchor.get('content', ''),
                            "source": anchor.get('source', ''),
                            "publish_date": anchor.get('publish_date', ''),
                            "url": anchor.get('url', '')
                        }
                    }
                else:
                    result["anchors"][dimension]["categories"][category] = {
                        "name": cat_name,
                        "status": "empty",
                        "document": None
                    }
        
        return result
    
    def generate_text_summary(self, anchors: Dict) -> str:
        """Generate a text summary for console display."""
        filled = self.count_filled_slots(anchors)
        
        # Calculate total slots based on dimension-specific categories
        total = 0
        for dimension in config.DIMENSIONS:
            if dimension in config.DIMENSION_CATEGORIES:
                total += len(config.DIMENSION_CATEGORIES[dimension])
            else:
                total += len(config.CATEGORIES)
        
        summary = f"""
Final Anchor Selection Summary
==============================
Total slots: {total}
Filled: {filled} ({filled/total*100:.1f}%)
Empty: {total - filled}

Detailed Status:
"""
        for dimension in config.DIMENSIONS:
            dim_name = config.DIMENSIONS[dimension]['name']
            summary += f"\n{dim_name}:\n"
            
            # Get categories for this dimension
            if dimension in config.DIMENSION_CATEGORIES:
                categories = config.DIMENSION_CATEGORIES[dimension]
            else:
                categories = config.CATEGORIES
            
            for category, cat_name in categories.items():
                if dimension in anchors and category in anchors[dimension]:
                    anchor = anchors[dimension][category]
                else:
                    anchor = None
                
                if anchor:
                    summary += f"  {cat_name}: ✓\n"
                    summary += f"    Doc ID: {anchor['doc_id']}\n"
                    if 'title' in anchor:
                        title = anchor['title'][:60] + "..." if len(anchor.get('title', '')) > 60 else anchor.get('title', '')
                        summary += f"    Title: {title}\n"
                else:
                    summary += f"  {cat_name}: ✗ (empty)\n"
        
        summary += f"\nTotal comparisons made: {self.total_comparisons}\n"
        
        return summary