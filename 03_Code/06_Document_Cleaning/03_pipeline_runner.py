#!/usr/bin/env python3
"""
Document Processing Pipeline Runner
Orchestrates the complete document processing pipeline:
1. Clean documents (remove navigation garbage)
2. Filter for carbon-related content
3. Generate comprehensive reports
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self):
        """Initialize pipeline runner"""
        self.base_path = Path(__file__).parent.parent.parent
        self.stats = {
            'start_time': datetime.now(),
            'stages': {}
        }
    
    def run_stage(self, stage_name: str, script_name: str) -> bool:
        """Run a single pipeline stage"""
        logger.info("=" * 70)
        logger.info(f"STAGE: {stage_name}")
        logger.info("=" * 70)
        
        script_path = Path(__file__).parent / script_name
        stage_start = datetime.now()
        
        try:
            # Run the script as a subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(script_path.parent)
            )
            
            if result.returncode != 0:
                logger.error(f"Stage '{stage_name}' failed with error:")
                logger.error(result.stderr)
                return False
            
            # Log output
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.stats['stages'][stage_name] = {
                'success': True,
                'duration_seconds': stage_duration,
                'timestamp': stage_start.isoformat()
            }
            
            logger.info(f"Stage completed in {stage_duration:.1f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed with exception: {e}")
            self.stats['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'timestamp': stage_start.isoformat()
            }
            return False
    
    def collect_statistics(self):
        """Collect statistics from both stages"""
        stats_summary = {}
        
        # Read cleaning statistics
        cleaning_stats_path = self.base_path / '02_Data_Processed' / '04_Documents_Cleaned' / 'cleaning_summary.json'
        if cleaning_stats_path.exists():
            with open(cleaning_stats_path, 'r', encoding='utf-8') as f:
                cleaning_stats = json.load(f)
                stats_summary['cleaning'] = {
                    'total_docs': cleaning_stats['total_docs'],
                    'docs_cleaned': cleaning_stats['docs_cleaned'],
                    'chars_removed': cleaning_stats['total_chars_removed']
                }
        
        # Read filtering statistics
        filter_stats_path = self.base_path / '02_Data_Processed' / '05_Policy_Doc_Filtered' / 'filter_summary.json'
        if filter_stats_path.exists():
            with open(filter_stats_path, 'r', encoding='utf-8') as f:
                filter_stats = json.load(f)
                stats_summary['filtering'] = {
                    'total_docs': filter_stats['summary']['total_original'],
                    'docs_retained': filter_stats['summary']['total_filtered'],
                    'retention_rate': filter_stats['summary']['overall_retention_rate']
                }
        
        return stats_summary
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        report_path = self.base_path / 'docs' / 'filtering' / 'pipeline_report.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.collect_statistics()
        total_duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Document Processing Pipeline Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Pipeline Execution Summary\n\n")
            f.write(f"- **Total Duration**: {total_duration:.1f} seconds\n")
            f.write(f"- **Stages Completed**: {len([s for s in self.stats['stages'].values() if s['success']])}/{len(self.stats['stages'])}\n\n")
            
            f.write("### Stage Details\n\n")
            f.write("| Stage | Status | Duration |\n")
            f.write("|-------|--------|----------|\n")
            for stage_name, stage_info in self.stats['stages'].items():
                status = "✅ Success" if stage_info['success'] else "❌ Failed"
                duration = f"{stage_info.get('duration_seconds', 0):.1f}s"
                f.write(f"| {stage_name} | {status} | {duration} |\n")
            
            f.write("\n## Data Flow Summary\n\n")
            
            if 'cleaning' in stats:
                f.write("### Stage 1: Document Cleaning\n")
                f.write(f"- **Input**: 3,312 raw documents\n")
                f.write(f"- **Documents Cleaned**: {stats['cleaning']['docs_cleaned']:,}\n")
                f.write(f"- **Characters Removed**: {stats['cleaning']['chars_removed']:,}\n")
                f.write(f"- **Output**: 3,312 cleaned documents\n\n")
            
            if 'filtering' in stats:
                f.write("### Stage 2: Carbon Filtering\n")
                f.write(f"- **Input**: {stats['filtering']['total_docs']:,} cleaned documents\n")
                f.write(f"- **Documents Retained**: {stats['filtering']['docs_retained']:,}\n")
                f.write(f"- **Retention Rate**: {stats['filtering']['retention_rate']:.1f}%\n")
                f.write(f"- **Documents Removed**: {stats['filtering']['total_docs'] - stats['filtering']['docs_retained']:,}\n\n")
            
            f.write("## Final Output\n\n")
            if 'filtering' in stats:
                f.write(f"- **Location**: `02_Data_Processed/05_Policy_Doc_Filtered/`\n")
                f.write(f"- **Total Documents**: {stats['filtering']['docs_retained']:,}\n")
                f.write(f"- **Reduction from Raw**: {((3312 - stats['filtering']['docs_retained']) / 3312 * 100):.1f}%\n\n")
            
            f.write("## Key Improvements\n\n")
            f.write("1. **Navigation Garbage Removed**: Cleaned GZETS headers/footers\n")
            f.write("2. **Carbon Focus**: Filtered out non-carbon environmental documents\n")
            f.write("3. **Size Reduction**: Smaller documents for NLP processing\n")
            f.write("4. **Cost Savings**: Reduced token count for API processing\n")
        
        logger.info(f"Pipeline report generated: {report_path}")
        return report_path
    
    def run_pipeline(self):
        """Run the complete document processing pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("DOCUMENT PROCESSING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Started at: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Stage 1: Clean documents
        if not self.run_stage("Document Cleaning", "01_clean_documents.py"):
            logger.error("Pipeline aborted due to cleaning stage failure")
            return False
        
        # Stage 2: Filter for carbon content
        if not self.run_stage("Carbon Filtering", "02_carbon_filter.py"):
            logger.error("Pipeline aborted due to filtering stage failure")
            return False
        
        # Generate final report
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING PIPELINE REPORT")
        logger.info("=" * 70)
        
        report_path = self.generate_pipeline_report()
        
        # Summary
        total_duration = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total duration: {total_duration:.1f} seconds")
        logger.info(f"Report saved to: {report_path}")
        
        # Display final statistics
        stats = self.collect_statistics()
        if 'filtering' in stats:
            logger.info(f"\nFinal output: {stats['filtering']['docs_retained']:,} documents")
            logger.info(f"Location: 02_Data_Processed/05_Policy_Doc_Filtered/")
        
        return True


def main():
    """Main execution function"""
    runner = PipelineRunner()
    success = runner.run_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()