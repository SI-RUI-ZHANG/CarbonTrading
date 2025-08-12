#!/usr/bin/env python3
"""
Master script to run all LSTM models with different configurations
Runs daily and weekly models for both GDEA and HBEA markets
With and without sentiment features where applicable
"""

import subprocess
import os
import json
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRunner:
    """Orchestrates running all LSTM model configurations"""
    
    def __init__(self):
        """Initialize the model runner"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results = {}
        self.start_time = datetime.now()
        
    def run_command(self, command: str, working_dir: str, description: str) -> Tuple[bool, str]:
        """
        Run a command and capture output
        
        Args:
            command: Command to run
            working_dir: Directory to run command in
            description: Description of what's being run
            
        Returns:
            Tuple of (success, output_message)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"Command: {command}")
        logger.info(f"Directory: {working_dir}")
        logger.info('='*60)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                # Try to extract key metrics from output
                metrics = self.extract_metrics(result.stdout)
                return True, metrics
            else:
                logger.error(f"❌ {description} failed")
                logger.error(f"Error: {result.stderr[:500]}")
                return False, result.stderr[:500]
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ {description} timed out after 30 minutes")
            return False, "Timeout"
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {str(e)}")
            return False, str(e)
    
    def extract_metrics(self, output: str) -> str:
        """Extract key metrics from model output"""
        lines = output.split('\n')
        metrics = []
        
        # Look for accuracy, loss, and other metrics
        for line in lines:
            if 'accuracy' in line.lower() or 'loss' in line.lower() or 'completed' in line.lower():
                metrics.append(line.strip())
        
        # Return last 5 relevant lines
        return '\n'.join(metrics[-5:]) if metrics else "No metrics found"
    
    def run_daily_lstm(self) -> Dict:
        """Run daily LSTM models (without sentiment)"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING DAILY LSTM MODELS (NO SENTIMENT)")
        logger.info("="*80)
        
        results = {}
        working_dir = os.path.join(self.base_dir, '10_LSTM_Daily')
        
        # Run for both markets (run.py handles both by default now)
        success, output = self.run_command(
            "python run.py",
            working_dir,
            "Daily LSTM for GDEA and HBEA"
        )
        
        results['both_markets'] = {
            'success': success,
            'output': output
        }
        
        return results
    
    def run_daily_lstm_sentiment(self) -> Dict:
        """Run daily LSTM models with sentiment features"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING DAILY LSTM MODELS WITH SENTIMENT")
        logger.info("="*80)
        
        results = {}
        working_dir = os.path.join(self.base_dir, '11_LSTM_Daily_Sentiment')
        
        # Run for both markets
        success, output = self.run_command(
            "python run.py",
            working_dir,
            "Daily LSTM with Sentiment for GDEA and HBEA"
        )
        
        results['both_markets'] = {
            'success': success,
            'output': output
        }
        
        return results
    
    def run_weekly_lstm(self) -> Dict:
        """Run weekly LSTM models with all configurations"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING WEEKLY LSTM MODELS")
        logger.info("="*80)
        
        results = {}
        working_dir = os.path.join(self.base_dir, '12_LSTM_Weekly')
        
        # Configuration combinations
        configs = [
            ("both", "both", "All markets with and without sentiment"),
            # Alternative: run specific combinations
            # ("GDEA", "yes", "GDEA with sentiment"),
            # ("GDEA", "no", "GDEA without sentiment"),
            # ("HBEA", "yes", "HBEA with sentiment"),
            # ("HBEA", "no", "HBEA without sentiment"),
        ]
        
        for market, sentiment, description in configs:
            command = f"python run.py --market {market} --sentiment {sentiment}"
            success, output = self.run_command(
                command,
                working_dir,
                f"Weekly LSTM: {description}"
            )
            
            key = f"{market}_{sentiment}"
            results[key] = {
                'success': success,
                'output': output,
                'description': description
            }
        
        return results
    
    def run_meta_models(self) -> Dict:
        """Run meta models (optional)"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING META MODELS (OPTIONAL)")
        logger.info("="*80)
        
        results = {}
        working_dir = os.path.join(self.base_dir, '13_Meta_Model')
        
        # Daily meta model
        success, output = self.run_command(
            "python run.py",
            working_dir,
            "Daily Meta Model for GDEA and HBEA"
        )
        
        results['daily_meta'] = {
            'success': success,
            'output': output
        }
        
        # Weekly meta model
        success, output = self.run_command(
            "python run_weekly.py --sentiment yes",
            working_dir,
            "Weekly Meta Model with Sentiment"
        )
        
        results['weekly_meta'] = {
            'success': success,
            'output': output
        }
        
        return results
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of all runs"""
        report_path = os.path.join(self.base_dir, f'model_run_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LSTM MODEL EXECUTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {datetime.now() - self.start_time}\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            total_runs = sum(len(v) for v in self.results.values())
            successful_runs = sum(1 for category in self.results.values() 
                                 for result in category.values() 
                                 if result.get('success', False))
            
            f.write(f"Total Configurations Run: {total_runs}\n")
            f.write(f"Successful: {successful_runs}\n")
            f.write(f"Failed: {total_runs - successful_runs}\n")
            f.write(f"Success Rate: {successful_runs/total_runs*100:.1f}%\n\n")
            
            # Detailed results
            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for category, results in self.results.items():
                f.write(f"\n{category.upper()}\n")
                f.write("-"*40 + "\n")
                
                for config, result in results.items():
                    status = "✅" if result.get('success', False) else "❌"
                    f.write(f"\n{status} {config}:\n")
                    if 'description' in result:
                        f.write(f"   Description: {result['description']}\n")
                    f.write(f"   Output: {result.get('output', 'No output')}\n")
        
        logger.info(f"\n📊 Summary report saved to: {report_path}")
        
        # Also print summary to console
        logger.info("\n" + "="*80)
        logger.info("EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Configurations Run: {total_runs}")
        logger.info(f"Successful: {successful_runs}")
        logger.info(f"Failed: {total_runs - successful_runs}")
        logger.info(f"Success Rate: {successful_runs/total_runs*100:.1f}%")
        logger.info(f"Total Runtime: {datetime.now() - self.start_time}")
    
    def run_all(self, include_meta: bool = False) -> None:
        """
        Run all models
        
        Args:
            include_meta: Whether to include meta models (default: False)
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE LSTM MODEL EXECUTION")
        logger.info(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        # Run each model category
        self.results['daily_lstm'] = self.run_daily_lstm()
        self.results['daily_lstm_sentiment'] = self.run_daily_lstm_sentiment()
        self.results['weekly_lstm'] = self.run_weekly_lstm()
        
        if include_meta:
            self.results['meta_models'] = self.run_meta_models()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL MODELS COMPLETED")
        logger.info("="*80)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all LSTM models with different configurations'
    )
    parser.add_argument(
        '--include-meta',
        action='store_true',
        help='Include meta models in the run (default: False)'
    )
    parser.add_argument(
        '--daily-only',
        action='store_true',
        help='Run only daily models (default: False)'
    )
    parser.add_argument(
        '--weekly-only',
        action='store_true',
        help='Run only weekly models (default: False)'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ModelRunner()
    
    # Determine what to run
    if args.daily_only:
        logger.info("Running DAILY models only")
        runner.results['daily_lstm'] = runner.run_daily_lstm()
        runner.results['daily_lstm_sentiment'] = runner.run_daily_lstm_sentiment()
        runner.generate_summary_report()
    elif args.weekly_only:
        logger.info("Running WEEKLY models only")
        runner.results['weekly_lstm'] = runner.run_weekly_lstm()
        runner.generate_summary_report()
    else:
        # Run all
        runner.run_all(include_meta=args.include_meta)


if __name__ == "__main__":
    main()