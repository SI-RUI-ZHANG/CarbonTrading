#!/usr/bin/env python3
"""
Document Cleaning Script
Removes navigation headers, footers, and other non-content elements from scraped documents
"""

import json
import re
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


class DocumentCleaner:
    def __init__(self, patterns_file: str = "cleaning_patterns.json"):
        """Initialize cleaner with pattern configuration"""
        self.base_path = Path(__file__).parent.parent.parent
        patterns_path = Path(__file__).parent / patterns_file
        
        with open(patterns_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.statistics = {
            'total_docs': 0,
            'docs_cleaned': 0,
            'total_chars_removed': 0,
            'patterns_applied': {}
        }
    
    def compile_patterns(self, source: str) -> List[Tuple[str, re.Pattern, str]]:
        """Compile regex patterns for a specific source"""
        compiled = []
        
        if source in self.config['sources'] and self.config['sources'][source]['enabled']:
            for pattern_config in self.config['sources'][source]['patterns']:
                flags = 0
                if 'DOTALL' in pattern_config.get('flags', []):
                    flags |= re.DOTALL
                if 'MULTILINE' in pattern_config.get('flags', []):
                    flags |= re.MULTILINE
                
                pattern = re.compile(pattern_config['pattern'], flags)
                compiled.append((
                    pattern_config['name'],
                    pattern,
                    pattern_config.get('replacement', '')
                ))
        
        # Add global patterns
        for pattern_config in self.config.get('global_patterns', []):
            pattern = re.compile(pattern_config['pattern'])
            compiled.append((
                pattern_config['name'],
                pattern,
                pattern_config.get('replacement', '')
            ))
        
        return compiled
    
    def clean_content(self, content: str, source: str) -> Tuple[str, int, List[str]]:
        """Clean content using appropriate patterns for the source"""
        original_length = len(content)
        patterns_used = []
        
        # Get compiled patterns for this source
        patterns = self.compile_patterns(source)
        
        # Apply each pattern
        for name, pattern, replacement in patterns:
            if pattern.search(content):
                content = pattern.sub(replacement, content)
                patterns_used.append(name)
                
                # Track pattern usage
                if name not in self.statistics['patterns_applied']:
                    self.statistics['patterns_applied'][name] = 0
                self.statistics['patterns_applied'][name] += 1
        
        # Clean up extra whitespace
        content = content.strip()
        
        chars_removed = original_length - len(content)
        return content, chars_removed, patterns_used
    
    def process_source(self, source: str, input_paths: Dict[str, str]) -> Dict[str, List[Dict]]:
        """Process all documents from a specific source"""
        cleaned_docs = {}
        
        for category, input_path in input_paths.items():
            full_input_path = self.base_path / input_path
            
            if not full_input_path.exists():
                logger.warning(f"File not found: {full_input_path}")
                continue
            
            logger.info(f"Processing {source}/{category}...")
            
            docs = []
            category_stats = {
                'total': 0,
                'cleaned': 0,
                'chars_removed': 0
            }
            
            with open(full_input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    category_stats['total'] += 1
                    self.statistics['total_docs'] += 1
                    
                    # Clean the content
                    if 'content' in doc and doc['content']:
                        cleaned_content, chars_removed, patterns_used = self.clean_content(
                            doc['content'], source
                        )
                        
                        if chars_removed > 0:
                            doc['_cleaning_metadata'] = {
                                'original_length': len(doc['content']),
                                'cleaned_length': len(cleaned_content),
                                'chars_removed': chars_removed,
                                'patterns_applied': patterns_used,
                                'cleaned_at': datetime.now().isoformat()
                            }
                            doc['content'] = cleaned_content
                            category_stats['cleaned'] += 1
                            category_stats['chars_removed'] += chars_removed
                            self.statistics['docs_cleaned'] += 1
                            self.statistics['total_chars_removed'] += chars_removed
                    
                    docs.append(doc)
            
            cleaned_docs[category] = docs
            
            # Log category statistics
            if category_stats['cleaned'] > 0:
                avg_removed = category_stats['chars_removed'] / category_stats['cleaned']
                logger.info(f"  Cleaned {category_stats['cleaned']}/{category_stats['total']} docs")
                logger.info(f"  Average chars removed: {avg_removed:.0f}")
            else:
                logger.info(f"  No cleaning needed for {category_stats['total']} docs")
        
        return cleaned_docs
    
    def save_cleaned_documents(self, source: str, category: str, documents: List[Dict]):
        """Save cleaned documents to output file"""
        output_dir = self.base_path / '02_Data_Processed' / '04_Documents_Cleaned' / source
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{category}_cleaned.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"  Saved to: {output_file}")
    
    def run_cleaning(self):
        """Run cleaning on all configured sources"""
        logger.info("Starting document cleaning...")
        logger.info("=" * 60)
        
        # Define input paths (same as filter config)
        input_structure = {
            "MEE": {
                "Decrees": "01_Data_Raw/03_Policy_Documents/MEE/Decrees/_all_documents.jsonl",
                "Notices": "01_Data_Raw/03_Policy_Documents/MEE/Notices/_all_documents.jsonl"
            },
            "HBETS": {
                "Center_Dynamics": "01_Data_Raw/03_Policy_Documents/HBETS/Center_Dynamics/_all_documents.jsonl"
            },
            "GZETS": {
                "Trading_Announcements": "01_Data_Raw/03_Policy_Documents/GZETS/Trading_Announcements/_all_documents.jsonl",
                "Center_Dynamics": "01_Data_Raw/03_Policy_Documents/GZETS/Center_Dynamics/_all_documents.jsonl",
                "Provincial_Municipal": "01_Data_Raw/03_Policy_Documents/GZETS/Provincial_Municipal/_all_documents.jsonl"
            }
        }
        
        # Process each source
        for source, categories in input_structure.items():
            cleaned_docs = self.process_source(source, categories)
            
            # Save cleaned documents
            for category, docs in cleaned_docs.items():
                self.save_cleaned_documents(source, category, docs)
        
        logger.info("=" * 60)
        logger.info("Cleaning complete!")
        logger.info(f"Total documents: {self.statistics['total_docs']}")
        logger.info(f"Documents cleaned: {self.statistics['docs_cleaned']}")
        logger.info(f"Total characters removed: {self.statistics['total_chars_removed']:,}")
        
        if self.statistics['docs_cleaned'] > 0:
            avg_removed = self.statistics['total_chars_removed'] / self.statistics['docs_cleaned']
            logger.info(f"Average chars removed per cleaned doc: {avg_removed:.0f}")
    
    def save_statistics(self):
        """Save cleaning statistics"""
        output_path = self.base_path / '02_Data_Processed' / '04_Documents_Cleaned' / 'cleaning_summary.json'
        
        stats = {
            **self.statistics,
            'timestamp': datetime.now().isoformat(),
            'patterns_config': self.config
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Statistics saved to: {output_path}")
    
    def generate_report(self):
        """Generate markdown report of cleaning results"""
        report_path = self.base_path / 'docs' / 'filtering' / 'document_cleaning_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Document Cleaning Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Documents Processed**: {self.statistics['total_docs']:,}\n")
            f.write(f"- **Documents Cleaned**: {self.statistics['docs_cleaned']:,}\n")
            f.write(f"- **Total Characters Removed**: {self.statistics['total_chars_removed']:,}\n")
            
            if self.statistics['docs_cleaned'] > 0:
                avg_removed = self.statistics['total_chars_removed'] / self.statistics['docs_cleaned']
                f.write(f"- **Average Characters Removed**: {avg_removed:.0f} per cleaned document\n\n")
            
            f.write("## Patterns Applied\n\n")
            f.write("| Pattern | Times Applied |\n")
            f.write("|---------|---------------|\n")
            for pattern, count in sorted(self.statistics['patterns_applied'].items(), 
                                        key=lambda x: x[1], reverse=True):
                f.write(f"| {pattern} | {count:,} |\n")
            
            f.write("\n## Impact Analysis\n\n")
            f.write("### GZETS Documents\n")
            f.write("- Primary source of navigation garbage\n")
            f.write("- Typical removal: 300-400 characters per document\n")
            f.write("- Patterns: navigation headers, footers, copyright info\n\n")
            
            f.write("### HBETS & MEE Documents\n")
            f.write("- Generally clean, minimal processing needed\n")
            f.write("- Only global whitespace cleaning applied\n")
        
        logger.info(f"Report generated: {report_path}")


def main():
    """Main execution function"""
    cleaner = DocumentCleaner()
    cleaner.run_cleaning()
    cleaner.save_statistics()
    cleaner.generate_report()


if __name__ == "__main__":
    main()