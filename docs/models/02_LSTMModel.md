# LSTM Direction Prediction Model

## Overview

The LSTM model predicts the **direction** of next-day carbon price movements (up vs down/flat) rather than exact returns. This binary classification approach addresses the fundamental challenge of financial time series: predicting direction is often more practical and achievable than predicting precise magnitudes.

**Key Features:**
- Handles class imbalance through BCEWithLogitsLoss with pos_weight
- Dynamic input size detection (49 features)
- 2-layer stacked LSTM architecture with dropout regularization
- Chronological train/validation/test splits to prevent data leakage

## Design Decisions

### 1. Classification Over Regression

**Approach:** Binary direction classification
- **Rationale:** Trading decisions primarily need direction, not exact magnitude
- **Implementation:** Converts log returns to binary labels
- **Benefit:** Clearer objective with well-defined success metrics

### 2. Binary vs Multi-Class

**Decision:** Two classes (Down/Flat vs Up)
- **Down/Flat (0):** log_return ≤ 0
- **Up (1):** log_return > 0

**Why not three classes (Down, Flat, Up)?**
- Carbon markets have many zero-return days (no trading)
- "Flat" is economically similar to "Down" for trading
- Binary simplifies the problem while maintaining practical value

### 3. Class Imbalance Handling

**Challenge:** Class imbalance in carbon market data
```
Typical Distribution:
  Down/Flat: ~70%
  Up:        ~30%
```

**Solution:** BCEWithLogitsLoss with pos_weight
```python
pos_weight = num_negatives / num_positives  # ≈ 2.54
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Solution Impact:**
- Prevents model from defaulting to majority class
- Encourages learning of minority class patterns
- Produces more balanced predictions

## Architecture

### Model Structure

```
Input (60 days × 49 features)
    ↓
LSTM Layer 1 (64 hidden units)
    ↓
LSTM Layer 2 (64 hidden units)
    ↓
Batch Normalization
    ↓
FC Layer 1 (64 → 64)
    ↓
FC Layer 2 (64 → 32)
    ↓
Output Layer (32 → 1)
    ↓
Sigmoid → Binary Prediction
```

### Key Components

**LSTM Layers:**
- 2-layer stacked LSTM with 64 hidden units
- Dropout (0.2) between layers for regularization
- Captures temporal dependencies in 60-day windows

**Fully Connected Layers:**
- Progressive dimension reduction (64 → 32 → 1)
- ReLU activation and dropout between layers
- Final layer outputs raw logit for BCEWithLogitsLoss

**Input Specifications:**
- Sequence length: 60 trading days
- Features: 49 (prices, volumes, technical indicators, macro data, temporal features)
- Dynamic input size detection from data

## Training Strategy

### Data Splits

Chronological splitting to prevent look-ahead bias:

| Split | Date Range | Purpose |
|-------|------------|---------|
| Train | 2014 → 2020 | Model training |
| Val | 2021 → 2022 | Hyperparameter tuning |
| Test | 2023+ | Final evaluation |

### Optimization

**Loss Function:**
- BCEWithLogitsLoss with pos_weight for class balancing
- Penalizes minority class errors more heavily

**Training Mechanics:**
- Adam optimizer (lr=0.001)
- ReduceLROnPlateau scheduler
- Early stopping (patience=15 epochs)
- Gradient clipping (max_norm=1.0)

**Data Handling:**
- MinMaxScaler fitted on training data only
- Configurable shuffle for training loader
- Batch size: 32

## Implementation Highlights

### Dynamic Configuration

```python
# Automatic input size detection
config.INPUT_SIZE = X_train.shape[2]  # Detects 49 features

# Timestamped experiment tracking
RUN_NAME = f"{timestamp}_{MARKET}_LSTM_Classification"
```

### Class Weight Calculation

The model automatically calculates appropriate class weights:

```python
num_negatives = np.sum(y_train == 0)
num_positives = np.sum(y_train == 1)
pos_weight = num_negatives / num_positives  # ≈ 2.54 for GDEA
```

### Prediction Logic

Binary predictions using sigmoid threshold:

```python
# Training/Evaluation
predicted = (torch.sigmoid(logits) > 0.5).float()

# Probability extraction for confidence analysis
probabilities = torch.sigmoid(logits)
```

## Model Evaluation

### Evaluation Metrics

The model is evaluated using standard binary classification metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: When predicting "up", how often is it correct?
- **Recall**: Of all actual "up" movements, how many are caught?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of true/false positives and negatives

### Class Balance Considerations

The pos_weight parameter in BCEWithLogitsLoss helps the model learn from both classes:
- Without balancing: Model may default to predicting majority class
- With balancing: Model learns patterns from both up and down movements
- Trade-off: May sacrifice some overall accuracy for better minority class detection

## Future Improvements

### Potential Enhancements

1. **Threshold Optimization**
   - Current: Fixed 0.5 threshold
   - Opportunity: Optimize threshold based on risk/reward preferences

2. **Ensemble Methods**
   - Combine multiple LSTM models with different architectures
   - Blend with traditional technical indicators

3. **Feature Engineering**
   - Add market microstructure features
   - Include cross-market signals (futures, related commodities)

4. **Alternative Architectures**
   - Attention mechanisms (Transformer-based)
   - CNN-LSTM hybrid for pattern recognition
   - GRU for potentially faster training

5. **Multi-Task Learning**
   - Jointly predict direction and volatility
   - Auxiliary tasks to improve representation learning

## Usage

### Running the Pipeline

```bash
# 1. Prepare data with direction labels
python data_preparation.py

# 2. Train classification model
python model_training.py

# 3. Evaluate performance
# Results saved to: 04_Models/{timestamp}_{MARKET}_LSTM_Classification/
```

### Configuration

Key parameters in `config.py`:
- `MARKET`: 'GDEA' or 'HBEA'
- `HIDDEN_SIZE`: 64 (LSTM hidden units)
- `NUM_LAYERS`: 2 (stacked LSTMs)
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001
- `EARLY_STOPPING_PATIENCE`: 15
- `SHUFFLE_TRAIN_LOADER`: True (configurable for ablation studies)

## Conclusion

The LSTM direction prediction model implements a binary classification approach for carbon price movement prediction. Key design decisions include:

1. **Binary Classification**: Focuses on direction (up vs down/flat) rather than exact returns
2. **Class Balancing**: Uses BCEWithLogitsLoss with pos_weight to handle imbalanced data
3. **Temporal Integrity**: Chronological splits prevent data leakage
4. **Dynamic Architecture**: Automatically adapts to feature count from data
5. **Regularization**: Multiple dropout layers and gradient clipping prevent overfitting

The model provides a foundation for carbon price direction prediction that can be extended with ensemble methods, alternative architectures, or additional feature engineering.