"""
Evaluation functions for LSTM classification model
Includes classification metrics and visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics"""
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    sensitivity = recall  # Same as recall (true positive rate)
    
    # Trading-specific metrics
    # Assuming 1 = up, 0 = down/flat
    # Precision for up predictions (important for trading)
    up_precision = precision  # How often we're right when we predict up
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'up_precision': up_precision,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics, cm

def evaluate_model(model, test_loader, config):
    """Evaluate classification model on test set"""
    model.eval()
    predictions = []
    actuals = []
    probabilities = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(config.DEVICE)
            
            # Get model predictions (logits)
            outputs = model(X_batch)
            
            # Convert to probabilities using sigmoid for binary classification
            probs = torch.sigmoid(outputs).squeeze()  # Remove extra dimension
            
            # Handle both single sample and batch cases
            if probs.dim() == 0:  # Single sample case
                probabilities.append(probs.cpu().item())
                predicted = (probs > 0.5).float()
                predictions.append(predicted.cpu().item())
                actuals.append(y_batch.item())
            else:  # Batch case
                probabilities.extend(probs.cpu().numpy())
                predicted = (probs > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    probabilities = np.array(probabilities)
    
    metrics, cm = calculate_metrics(actuals, predictions)
    
    return predictions, actuals, metrics, cm, probabilities

def plot_predictions(predictions, actuals, metrics, cm, probabilities, config):
    """Plot classification results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title(f'{config.MARKET} - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticklabels(['Down/Flat', 'Up'])
    axes[0, 0].set_yticklabels(['Down/Flat', 'Up'])
    
    # 2. Prediction Distribution
    axes[0, 1].hist([probabilities[actuals==0], probabilities[actuals==1]], 
                    bins=30, label=['Actual Down/Flat', 'Actual Up'], 
                    alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Probability Distribution by True Class')
    axes[0, 1].set_xlabel('Predicted Probability of Up')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time Series of Predictions (first 200 points)
    n_points = min(200, len(predictions))
    x_axis = np.arange(n_points)
    
    # Create a plot showing actual vs predicted over time
    axes[0, 2].scatter(x_axis[actuals[:n_points]==0], 
                      np.zeros(sum(actuals[:n_points]==0)), 
                      color='red', alpha=0.5, label='Actual Down/Flat', s=20)
    axes[0, 2].scatter(x_axis[actuals[:n_points]==1], 
                      np.ones(sum(actuals[:n_points]==1)), 
                      color='green', alpha=0.5, label='Actual Up', s=20)
    axes[0, 2].scatter(x_axis, predictions[:n_points] + 0.05, 
                      color='blue', alpha=0.3, label='Predicted', s=10, marker='^')
    axes[0, 2].set_title(f'Predictions vs Actuals (First {n_points} points)')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Direction (0=Down/Flat, 1=Up)')
    axes[0, 2].set_yticks([0, 1])
    axes[0, 2].set_yticklabels(['Down/Flat', 'Up'])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cumulative Accuracy Over Time
    rolling_accuracy = []
    window = 50
    for i in range(window, len(predictions)):
        acc = np.mean(predictions[i-window:i] == actuals[i-window:i])
        rolling_accuracy.append(acc)
    
    axes[1, 0].plot(rolling_accuracy, alpha=0.7)
    axes[1, 0].axhline(metrics['accuracy'], color='red', linestyle='--', 
                       label=f'Overall Accuracy: {metrics["accuracy"]:.3f}')
    axes[1, 0].set_title(f'Rolling Accuracy (window={window})')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Classification Report as Text
    axes[1, 1].axis('off')
    report_text = f"""
    Classification Metrics:
    ─────────────────────────
    Accuracy:    {metrics['accuracy']:.3f}
    Precision:   {metrics['precision']:.3f}
    Recall:      {metrics['recall']:.3f}
    F1-Score:    {metrics['f1_score']:.3f}
    
    Confusion Matrix Details:
    ─────────────────────────
    True Positives:  {metrics['true_positives']}
    True Negatives:  {metrics['true_negatives']}
    False Positives: {metrics['false_positives']}
    False Negatives: {metrics['false_negatives']}
    
    Trading Metrics:
    ─────────────────────────
    Up Precision: {metrics['up_precision']:.3f}
    (When we predict Up, we're right {metrics['up_precision']*100:.1f}% of the time)
    """
    axes[1, 1].text(0.1, 0.5, report_text, fontsize=10, family='monospace',
                   verticalalignment='center')
    axes[1, 1].set_title('Performance Summary')
    
    # 6. Prediction Confidence Distribution
    axes[1, 2].hist(probabilities, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
    axes[1, 2].set_title('Prediction Confidence Distribution')
    axes[1, 2].set_xlabel('Probability of Up Movement')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add confidence bands
    high_conf = probabilities[(probabilities > 0.7) | (probabilities < 0.3)]
    axes[1, 2].axvspan(0.7, 1.0, alpha=0.1, color='green', label='High Confidence Up')
    axes[1, 2].axvspan(0.0, 0.3, alpha=0.1, color='red', label='High Confidence Down')
    
    plt.suptitle(f'{config.MARKET} - Classification Results Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/classification_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()