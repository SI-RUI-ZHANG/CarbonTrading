"""
Visualization utilities for Weekly LSTM model
Generates training history and classification analysis plots
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_history(history: dict, config, save_path: str = None):
    """
    Plot training history for classification
    
    Args:
        history: Dictionary with train_loss, val_loss, val_accuracy lists
        config: Configuration object
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Binary CrossEntropy Loss')
    plt.title(f'{config.MARKET} Weekly - Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy subplot
    plt.subplot(1, 3, 2)
    # Convert to percentage for display
    val_acc_pct = [acc * 100 for acc in history['val_accuracy']]
    plt.plot(val_acc_pct, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{config.MARKET} Weekly - Accuracy History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed accuracy (last 2/3)
    plt.subplot(1, 3, 3)
    if len(val_acc_pct) > 3:
        start = len(val_acc_pct) // 3
        plt.plot(range(start, len(val_acc_pct)), val_acc_pct[start:], 
                label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{config.MARKET} Weekly - Accuracy (Last 2/3)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close()


def create_classification_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray, metrics: dict, 
                                  config, save_path: str = None):
    """
    Create comprehensive classification analysis visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        metrics: Dictionary with performance metrics
        config: Configuration object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticklabels(['Down/Flat', 'Up'])
    axes[0, 0].set_yticklabels(['Down/Flat', 'Up'])
    
    # 2. Prediction Distribution
    axes[0, 1].hist([y_true, y_pred], label=['Actual', 'Predicted'], 
                    bins=2, alpha=0.7, color=['blue', 'orange'])
    axes[0, 1].set_title('Class Distribution')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks([0.25, 0.75])
    axes[0, 1].set_xticklabels(['Down/Flat', 'Up'])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Weekly Prediction Pattern (Time Series View)
    axes[0, 2].plot(y_true, 'o-', label='Actual', alpha=0.6)
    axes[0, 2].plot(y_pred, 's-', label='Predicted', alpha=0.6)
    axes[0, 2].set_title('Weekly Predictions Over Time')
    axes[0, 2].set_xlabel('Week Index')
    axes[0, 2].set_ylabel('Direction (0=Down, 1=Up)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Prediction Accuracy by Confidence
    high_conf_mask = (y_prob > 0.7) | (y_prob < 0.3)
    mid_conf_mask = ~high_conf_mask
    
    if high_conf_mask.sum() > 0:
        high_conf_acc = (y_pred[high_conf_mask] == y_true[high_conf_mask]).mean()
    else:
        high_conf_acc = 0
        
    if mid_conf_mask.sum() > 0:
        mid_conf_acc = (y_pred[mid_conf_mask] == y_true[mid_conf_mask]).mean()
    else:
        mid_conf_acc = 0
    
    conf_data = {
        'High\nConfidence': high_conf_acc,
        'Medium\nConfidence': mid_conf_acc,
        'Overall': metrics.get('test_accuracy', metrics.get('accuracy', 0))
    }
    
    bars = axes[1, 0].bar(conf_data.keys(), conf_data.values(), 
                          color=['green', 'yellow', 'blue'], alpha=0.7)
    axes[1, 0].set_title('Accuracy by Confidence Level')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, conf_data.values()):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2%}', ha='center', va='bottom')
    
    # 5. Performance Summary (Text)
    sentiment_tag = 'With Sentiment' if config.USE_SENTIMENT else 'No Sentiment'
    report_text = f"""
    WEEKLY LSTM - {config.MARKET} ({sentiment_tag})
    ═══════════════════════════════════════════
    
    Classification Metrics:
    ─────────────────────────
    Accuracy:    {metrics.get('test_accuracy', metrics.get('accuracy', 0)):.3f}
    Precision:   {metrics.get('test_precision', metrics.get('precision', 0)):.3f}
    Recall:      {metrics.get('test_recall', metrics.get('recall', 0)):.3f}
    F1-Score:    {metrics.get('test_f1', metrics.get('f1_score', 0)):.3f}
    
    Confusion Matrix Details:
    ─────────────────────────
    True Positives:  {metrics['true_positives']}
    True Negatives:  {metrics['true_negatives']}
    False Positives: {metrics['false_positives']}
    False Negatives: {metrics['false_negatives']}
    
    Weekly Trading Insights:
    ─────────────────────────
    Up Weeks Detected: {metrics.get('test_recall', metrics.get('recall', 0))*100:.1f}%
    Up Precision: {metrics.get('test_precision', metrics.get('precision', 0))*100:.1f}%
    """
    axes[1, 1].text(0.1, 0.5, report_text, fontsize=9, family='monospace',
                   verticalalignment='center')
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    # 6. Prediction Confidence Distribution
    axes[1, 2].hist(y_prob, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 2].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
    axes[1, 2].set_title('Prediction Confidence Distribution')
    axes[1, 2].set_xlabel('Probability of Up Movement')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add confidence bands
    axes[1, 2].axvspan(0.7, 1.0, alpha=0.1, color='green')
    axes[1, 2].axvspan(0.0, 0.3, alpha=0.1, color='red')
    axes[1, 2].text(0.85, axes[1, 2].get_ylim()[1]*0.9, 'High\nConf Up', 
                   ha='center', fontsize=8)
    axes[1, 2].text(0.15, axes[1, 2].get_ylim()[1]*0.9, 'High\nConf Down', 
                   ha='center', fontsize=8)
    
    plt.suptitle(f'{config.MARKET} Weekly LSTM - Classification Results Analysis', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_config(config, output_path: str):
    """
    Save configuration to JSON for reproducibility
    
    Args:
        config: Configuration object
        output_path: Path to save config.json
    """
    import json
    
    config_dict = {
        'MARKET': config.MARKET,
        'RUN_NAME': config.run_name,
        'TASK_TYPE': 'Weekly Binary Classification',
        'USE_SENTIMENT': config.USE_SENTIMENT,
        'AGGREGATION': config.AGGREGATION,
        'SEQUENCE_LENGTH': config.SEQUENCE_LENGTH,
        'HIDDEN_SIZE': config.HIDDEN_SIZE,
        'NUM_LAYERS': config.NUM_LAYERS,
        'DROPOUT': config.DROPOUT,
        'BATCH_SIZE': config.BATCH_SIZE,
        'LEARNING_RATE': config.LEARNING_RATE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'EARLY_STOPPING_PATIENCE': config.EARLY_STOPPING_PATIENCE,
        'MIN_DAYS_PER_WEEK': config.MIN_DAYS_PER_WEEK,
        'SHUFFLE_TRAIN_LOADER': config.SHUFFLE_TRAIN_LOADER,
        'SEED': config.SEED,
        'DEVICE': str(config.DEVICE),
        'TRAIN_END_DATE': config.TRAIN_END_DATE,
        'VAL_END_DATE': config.VAL_END_DATE
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)