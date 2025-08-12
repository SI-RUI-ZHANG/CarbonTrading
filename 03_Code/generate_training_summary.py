#!/usr/bin/env python3
"""
Generate comprehensive summary of all LSTM model training results
"""

import os
import json
import pandas as pd
from datetime import datetime
import glob

def extract_metrics_from_log(log_file):
    """Extract key metrics from a training log file"""
    metrics = {
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'epochs_trained': None,
        'best_val_acc': None,
        'training_time': None
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if 'accuracy' in line.lower() and ':' in line:
                try:
                    if 'test' in line.lower() or 'final' in line.lower():
                        val = line.split(':')[-1].strip()
                        if val.replace('.', '').replace('-', '').isdigit():
                            metrics['accuracy'] = float(val)
                except:
                    pass
                    
            if 'precision' in line.lower() and ':' in line:
                try:
                    val = line.split(':')[-1].strip()
                    if val.replace('.', '').replace('-', '').isdigit():
                        metrics['precision'] = float(val)
                except:
                    pass
                    
            if 'recall' in line.lower() and ':' in line:
                try:
                    val = line.split(':')[-1].strip()
                    if val.replace('.', '').replace('-', '').isdigit():
                        metrics['recall'] = float(val)
                except:
                    pass
                    
            if 'f1_score' in line.lower() and ':' in line:
                try:
                    val = line.split(':')[-1].strip()
                    if val.replace('.', '').replace('-', '').isdigit():
                        metrics['f1_score'] = float(val)
                except:
                    pass
                    
            if 'epoch' in line.lower() and '/' in line:
                try:
                    epoch_part = line.split('Epoch')[-1].split('/')[0].strip()
                    if epoch_part.isdigit():
                        metrics['epochs_trained'] = int(epoch_part)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        
    return metrics

def find_model_directories():
    """Find all model directories created today"""
    base_dir = "/Users/siruizhang/Desktop/碳交易/Project/04_Models"
    today = datetime.now().strftime("%Y%m%d")
    
    model_dirs = []
    
    # Check daily models
    daily_dirs = glob.glob(f"{base_dir}/daily/*/*")
    for d in daily_dirs:
        if os.path.isdir(d):
            model_dirs.append(('Daily', d))
            
    # Check weekly models  
    weekly_dirs = glob.glob(f"{base_dir}/weekly/*/*")
    for d in weekly_dirs:
        if os.path.isdir(d):
            model_dirs.append(('Weekly', d))
            
    # Check meta models
    meta_dirs = glob.glob(f"{base_dir}/meta/*/*")
    for d in meta_dirs:
        if os.path.isdir(d):
            model_dirs.append(('Meta', d))
            
    # Check timestamped models
    timestamped_dirs = glob.glob(f"{base_dir}/{today}*")
    for d in timestamped_dirs:
        if os.path.isdir(d):
            model_dirs.append(('Timestamped', d))
            
    return model_dirs

def load_model_metrics(model_dir):
    """Load metrics.json from a model directory"""
    metrics_file = os.path.join(model_dir, 'metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def generate_summary_report():
    """Generate comprehensive summary of all training results"""
    
    print("="*80)
    print("LSTM MODEL TRAINING SUMMARY REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Find latest log directory
    log_dirs = glob.glob("/Users/siruizhang/Desktop/碳交易/Project/03_Code/training_logs_*")
    if log_dirs:
        latest_log_dir = max(log_dirs)
        print(f"Latest training logs: {latest_log_dir}")
        print()
        
        # Process each log file
        log_files = glob.glob(f"{latest_log_dir}/*.log")
        
        print("TRAINING RESULTS FROM LOGS:")
        print("-"*60)
        
        for log_file in sorted(log_files):
            log_name = os.path.basename(log_file).replace('.log', '')
            metrics = extract_metrics_from_log(log_file)
            
            print(f"\n{log_name}:")
            if metrics['accuracy'] is not None:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if metrics['precision'] is not None:
                print(f"  Precision: {metrics['precision']:.4f}")
            if metrics['recall'] is not None:
                print(f"  Recall: {metrics['recall']:.4f}")
            if metrics['f1_score'] is not None:
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
            if metrics['epochs_trained'] is not None:
                print(f"  Epochs Trained: {metrics['epochs_trained']}")
    
    print()
    print("="*80)
    print("MODEL DIRECTORIES AND SAVED METRICS:")
    print("-"*60)
    
    # Find and report on model directories
    model_dirs = find_model_directories()
    
    if model_dirs:
        for model_type, model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            print(f"\n[{model_type}] {model_name}:")
            print(f"  Path: {model_dir}")
            
            metrics = load_model_metrics(model_dir)
            if metrics:
                if 'test_metrics' in metrics:
                    test_metrics = metrics['test_metrics']
                    print(f"  Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
                    print(f"  Test F1: {test_metrics.get('f1_score', 'N/A')}")
                elif 'accuracy' in metrics:
                    print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
                elif 'validation' in metrics:
                    val_metrics = metrics['validation']
                    print(f"  Validation Accuracy: {val_metrics.get('accuracy', 'N/A')}")
            
            # Check for saved model files
            model_files = glob.glob(f"{model_dir}/*.pth") + glob.glob(f"{model_dir}/*.pkl")
            if model_files:
                print(f"  Saved Models: {len(model_files)} files")
    
    print()
    print("="*80)
    print("SUMMARY STATISTICS:")
    print("-"*60)
    
    # Count successful trainings
    if log_dirs:
        successful = 0
        failed = 0
        
        for log_file in glob.glob(f"{latest_log_dir}/*.log"):
            with open(log_file, 'r') as f:
                content = f.read()
                if 'completed successfully' in content.lower() or 'all markets completed' in content.lower():
                    successful += 1
                elif 'error' in content.lower() or 'failed' in content.lower():
                    failed += 1
        
        print(f"Total Models Trained: {successful + failed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        if successful + failed > 0:
            print(f"Success Rate: {successful/(successful+failed)*100:.1f}%")
    
    print()
    print("="*80)
    print("✅ SUMMARY REPORT COMPLETED")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report()