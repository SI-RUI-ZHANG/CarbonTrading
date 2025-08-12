"""
Performance Tracker for Carbon Price Prediction Models
Maintains a consolidated performance summary for all models
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from pathlib import Path


class PerformanceTracker:
    """Manages the central performance summary for all models"""
    
    # Model type mappings for display
    MODEL_DISPLAY_NAMES = {
        'daily_base': 'Daily Base',
        'daily_sentiment': 'Daily Sentiment',
        'weekly_base': 'Weekly Base',
        'weekly_sentiment': 'Weekly Sentiment',
        'daily_meta': 'Daily Meta',
        'weekly_meta': 'Weekly Meta'
    }
    
    def __init__(self, summary_path: str = '../../04_Models/performance_summary.txt'):
        """
        Initialize performance tracker
        
        Args:
            summary_path: Path to the performance summary file
        """
        self.summary_path = Path(summary_path)
        self.data = self._load_existing_data()
        
    def _load_existing_data(self) -> Dict:
        """Load existing performance data if available"""
        json_path = self.summary_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def update_performance(self, model_type: str, market: str, metrics: Dict, 
                          features_type: str = 'Base'):
        """
        Update performance for a specific model
        
        Args:
            model_type: Type of model (daily_base, weekly_sentiment, etc.)
            market: Market name (GDEA or HBEA)
            metrics: Dictionary containing performance metrics
            features_type: Type of features used (Base, +Sentiment, etc.)
        """
        # Create unique key for this model
        key = f"{model_type}_{market}"
        
        # Extract metrics with safe defaults
        entry = {
            'model_type': model_type,
            'model_display': self.MODEL_DISPLAY_NAMES.get(model_type, model_type),
            'market': market,
            'features': features_type,
            'accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 0)),
            'precision': metrics.get('test_precision', metrics.get('precision', 0)),
            'recall': metrics.get('test_recall', metrics.get('recall', 0)),
            'f1_score': metrics.get('test_f1', metrics.get('f1_score', 0)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add meta-specific metrics if applicable
        if 'meta' in model_type:
            entry['coverage'] = metrics.get('coverage', 0)
            entry['abstention_rate'] = metrics.get('abstention_rate', 0)
            entry['trades_made'] = metrics.get('trades_made', 0)
        
        # Update data
        self.data[key] = entry
        
        # Save both JSON and TXT formats
        self._save_json()
        self._save_txt()
        
    def _save_json(self):
        """Save data in JSON format for easy loading"""
        json_path = self.summary_path.with_suffix('.json')
        os.makedirs(json_path.parent, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def _save_txt(self):
        """Save human-readable text summary"""
        os.makedirs(self.summary_path.parent, exist_ok=True)
        
        with open(self.summary_path, 'w') as f:
            # Header
            f.write("=" * 100 + "\n")
            f.write("CARBON PRICE PREDICTION MODEL PERFORMANCE SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # Organize models by category
            daily_models = {k: v for k, v in self.data.items() if 'daily' in k and 'meta' not in k}
            weekly_models = {k: v for k, v in self.data.items() if 'weekly' in k and 'meta' not in k}
            meta_models = {k: v for k, v in self.data.items() if 'meta' in k}
            
            # Get base model F1 scores for comparison
            base_f1_scores = {}
            for key, model in self.data.items():
                if 'daily_base' in key or 'weekly_base' in key:
                    market = model.get('market')
                    model_type = 'daily' if 'daily' in key else 'weekly'
                    base_f1_scores[f"{model_type}_{market}"] = model.get('f1_score', 0)
            
            # Daily Models Section
            if daily_models:
                f.write("DAILY MODELS\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Model Type':<20} | {'Market':<6} | {'Features':<10} | "
                       f"{'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | "
                       f"{'F1':>6} | {'F1 Gain':>8}\n")
                f.write("-" * 100 + "\n")
                
                for key in sorted(daily_models.keys()):
                    m = daily_models[key]
                    
                    # Calculate F1 gain for sentiment models
                    if 'sentiment' in key:
                        base_key = f"daily_{m['market']}"
                        base_f1 = base_f1_scores.get(base_key, 0)
                        if base_f1 > 0:
                            f1_gain = ((m['f1_score'] - base_f1) / base_f1) * 100
                            f1_gain_str = f"{f1_gain:+.1f}%"
                        else:
                            f1_gain_str = "N/A"
                    else:
                        f1_gain_str = "baseline"
                    
                    f.write(f"{m['model_display']:<20} | {m['market']:<6} | {m['features']:<10} | "
                           f"{m['accuracy']:>8.3f} | {m['precision']:>9.3f} | {m['recall']:>7.3f} | "
                           f"{m['f1_score']:>6.3f} | {f1_gain_str:>8}\n")
                f.write("\n")
            
            # Weekly Models Section
            if weekly_models:
                f.write("WEEKLY MODELS\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Model Type':<20} | {'Market':<6} | {'Features':<10} | "
                       f"{'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | "
                       f"{'F1':>6} | {'F1 Gain':>8}\n")
                f.write("-" * 100 + "\n")
                
                for key in sorted(weekly_models.keys()):
                    m = weekly_models[key]
                    
                    # Calculate F1 gain for sentiment models
                    if 'sentiment' in key:
                        base_key = f"weekly_{m['market']}"
                        base_f1 = base_f1_scores.get(base_key, 0)
                        if base_f1 > 0:
                            f1_gain = ((m['f1_score'] - base_f1) / base_f1) * 100
                            f1_gain_str = f"{f1_gain:+.1f}%"
                        else:
                            f1_gain_str = "N/A"
                    else:
                        f1_gain_str = "baseline"
                    
                    f.write(f"{m['model_display']:<20} | {m['market']:<6} | {m['features']:<10} | "
                           f"{m['accuracy']:>8.3f} | {m['precision']:>9.3f} | {m['recall']:>7.3f} | "
                           f"{m['f1_score']:>6.3f} | {f1_gain_str:>8}\n")
                f.write("\n")
            
            # Meta Models Section
            if meta_models:
                f.write("META MODELS (100% Coverage - No Abstention)\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Model Type':<20} | {'Market':<6} | "
                       f"{'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | "
                       f"{'F1':>6} | {'F1 Gain':>8}\n")
                f.write("-" * 100 + "\n")
                
                for key in sorted(meta_models.keys()):
                    m = meta_models[key]
                    coverage = m.get('coverage', 0)
                    
                    # Format accuracy/precision/recall based on coverage
                    if coverage > 0:
                        acc_str = f"{m['accuracy']:.3f}"
                        prec_str = f"{m['precision']:.3f}"
                        rec_str = f"{m['recall']:.3f}"
                        f1_str = f"{m['f1_score']:.3f}"
                        
                        # Calculate F1 gain
                        model_type = 'daily' if 'daily' in key else 'weekly'
                        base_key = f"{model_type}_{m['market']}"
                        base_f1 = base_f1_scores.get(base_key, 0)
                        if base_f1 > 0:
                            f1_gain = ((m['f1_score'] - base_f1) / base_f1) * 100
                            f1_gain_str = f"{f1_gain:+.1f}%"
                        else:
                            f1_gain_str = "N/A"
                    else:
                        acc_str = "N/A"
                        prec_str = "N/A"
                        rec_str = "N/A"
                        f1_str = "N/A"
                        f1_gain_str = "N/A"
                    
                    f.write(f"{m['model_display']:<20} | {m['market']:<6} | "
                           f"{acc_str:>8} | {prec_str:>9} | {rec_str:>7} | "
                           f"{f1_str:>6} | {f1_gain_str:>8}\n")
                
                f.write("\n* Accuracy/Precision/Recall/F1 calculated only on samples where model made predictions (not abstained)\n\n")
            
            # Footer with legend
            f.write("Legend:\n")
            f.write("- Base: Models without sentiment features\n")
            f.write("- +Sentiment: Models with sentiment features included\n")
            f.write("- F1 Gain: Percentage improvement in F1 score vs base model\n")
            f.write("- Meta Models: Error reversal approach with 100% coverage (no abstention)\n")
            f.write("\n" + "=" * 100 + "\n")


def update_performance_summary(model_type: str, market: str, metrics: Dict, 
                              features_type: str = 'Base', 
                              summary_path: str = '../../04_Models/performance_summary.txt'):
    """
    Convenience function to update performance summary
    
    Args:
        model_type: Type of model (daily_base, weekly_sentiment, etc.)
        market: Market name (GDEA or HBEA)
        metrics: Dictionary containing performance metrics
        features_type: Type of features used (Base, +Sentiment, etc.)
        summary_path: Path to the performance summary file
    """
    tracker = PerformanceTracker(summary_path)
    tracker.update_performance(model_type, market, metrics, features_type)


# Create an empty __init__.py to make utils a package
def create_init_file():
    """Create __init__.py file for utils package"""
    init_path = Path(__file__).parent / '__init__.py'
    if not init_path.exists():
        init_path.write_text('"""Utils package for carbon price prediction models"""')


# Auto-create __init__.py when module is imported
create_init_file()