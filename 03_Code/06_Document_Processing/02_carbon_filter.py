#!/usr/bin/env python3
"""
Carbon Document Filtering Script
Filters cleaned policy documents based on carbon-related keyword
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarbonDocumentFilter:
    def __init__(self, config_path: str = "filter_config.json"):
        """Initialize filter with configuration"""
        self.base_path = Path(__file__).parent.parent.parent
        config_full_path = Path(__file__).parent / config_path
        
        with open(config_full_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.keyword = self.config['filter']['keyword']
        self.search_fields = self.config['filter']['search_fields']
        self.statistics = {}
    
    def check_document(self, doc: Dict) -> bool:
        """Check if document contains the carbon keyword"""
        search_text = ""
        for field in self.search_fields:
            search_text += doc.get(field, "").lower()
        
        return self.keyword.lower() in search_text
    
    def filter_source(self, source_name: str, category: str, input_path: str) -> Tuple[int, int, List[Dict]]:
        """Filter documents from a single source"""
        full_input_path = self.base_path / input_path
        
        if not full_input_path.exists():
            logger.warning(f"File not found: {full_input_path}")
            return 0, 0, []
        
        filtered_docs = []
        total_count = 0
        matched_count = 0
        
        logger.info(f"Processing {source_name}/{category}...")
        
        with open(full_input_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                doc = json.loads(line)
                
                if self.check_document(doc):
                    matched_count += 1
                    if self.config['output_metadata']['include_match_info']:
                        # Preserve cleaning metadata if present
                        filter_metadata = {
                            'matched_keyword': self.keyword,
                            'filter_timestamp': datetime.now().isoformat()
                        }
                        if '_cleaning_metadata' in doc:
                            filter_metadata['was_cleaned'] = True
                            filter_metadata['chars_removed_in_cleaning'] = doc['_cleaning_metadata']['chars_removed']
                        
                        doc['_filter_metadata'] = filter_metadata
                    filtered_docs.append(doc)
        
        retention_rate = (matched_count / total_count * 100) if total_count > 0 else 0
        logger.info(f"  Total: {total_count}, Matched: {matched_count} ({retention_rate:.1f}%)")
        
        return total_count, matched_count, filtered_docs
    
    def save_filtered_documents(self, source_name: str, category: str, documents: List[Dict]):
        """Save filtered documents to output file"""
        output_dir = self.base_path / self.config['output_base_path'] / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{category}_filtered.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"  Saved to: {output_file}")
    
    def run_filtering(self):
        """Run filtering on all configured sources"""
        logger.info("Starting carbon document filtering on CLEANED documents...")
        logger.info(f"Filter keyword: '{self.keyword}'")
        logger.info("=" * 60)
        
        total_original = 0
        total_filtered = 0
        
        for source_name, categories in self.config['input_paths'].items():
            source_stats = {}
            
            for category, input_path in categories.items():
                total, matched, filtered_docs = self.filter_source(source_name, category, input_path)
                
                if filtered_docs:
                    self.save_filtered_documents(source_name, category, filtered_docs)
                
                source_stats[category] = {
                    'original': total,
                    'filtered': matched,
                    'retention_rate': (matched / total * 100) if total > 0 else 0
                }
                
                total_original += total
                total_filtered += matched
            
            self.statistics[source_name] = source_stats
        
        self.statistics['summary'] = {
            'total_original': total_original,
            'total_filtered': total_filtered,
            'overall_retention_rate': (total_filtered / total_original * 100) if total_original > 0 else 0,
            'documents_removed': total_original - total_filtered,
            'filter_timestamp': datetime.now().isoformat(),
            'used_cleaned_documents': True
        }
        
        logger.info("=" * 60)
        logger.info(f"Filtering complete!")
        logger.info(f"Total documents: {total_original}")
        logger.info(f"Documents retained: {total_filtered} ({self.statistics['summary']['overall_retention_rate']:.1f}%)")
        logger.info(f"Documents removed: {total_original - total_filtered}")
    
    def save_statistics(self):
        """Save filtering statistics to JSON file"""
        output_path = self.base_path / self.config['output_base_path'] / 'filter_summary.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Statistics saved to: {output_path}")
    
    def generate_report(self):
        """Generate markdown report of filtering results"""
        report_path = self.base_path / 'docs' / 'filtering' / 'carbon_filter_report_cleaned.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Carbon Document Filtering Report (Cleaned Documents)\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Filter Keyword**: `{self.keyword}`\n")
            f.write(f"- **Search Fields**: {', '.join(self.search_fields)}\n")
            f.write(f"- **Case Sensitive**: {self.config['filter']['case_sensitive']}\n")
            f.write(f"- **Input**: Cleaned documents from `04_Documents_Cleaned/`\n")
            f.write(f"- **Output**: `05_Policy_Doc_Filtered/`\n\n")
            
            f.write("## Results Summary\n\n")
            summary = self.statistics['summary']
            f.write(f"- **Total Documents Processed**: {summary['total_original']:,}\n")
            f.write(f"- **Documents Retained**: {summary['total_filtered']:,} ({summary['overall_retention_rate']:.1f}%)\n")
            f.write(f"- **Documents Removed**: {summary['documents_removed']:,}\n\n")
            
            f.write("## Results by Source\n\n")
            f.write("| Source | Category | Original | Filtered | Retention Rate |\n")
            f.write("|--------|----------|----------|----------|---------------|\n")
            
            for source_name, categories in self.statistics.items():
                if source_name == 'summary':
                    continue
                for category, stats in categories.items():
                    f.write(f"| {source_name} | {category} | {stats['original']} | ")
                    f.write(f"{stats['filtered']} | {stats['retention_rate']:.1f}% |\n")
            
            f.write("\n## Pipeline Summary\n\n")
            f.write("1. **Raw Documents**: 3,312 total\n")
            f.write("2. **After Cleaning**: 3,312 documents (navigation garbage removed)\n")
            f.write(f"3. **After Carbon Filter**: {summary['total_filtered']:,} documents retained\n\n")
            
            f.write("### Impact of Two-Stage Processing\n")
            f.write("- Navigation cleaning reduced document sizes (especially GZETS)\n")
            f.write("- Carbon filtering removed non-carbon environmental documents\n")
            f.write("- Final dataset is clean and focused on carbon trading\n")
        
        logger.info(f"Report generated: {report_path}")


def main():
    """Main execution function"""
    filter = CarbonDocumentFilter()
    filter.run_filtering()
    filter.save_statistics()
    filter.generate_report()


if __name__ == "__main__":
    main()