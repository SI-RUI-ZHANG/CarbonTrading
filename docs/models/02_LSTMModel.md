# LSTM Direction Prediction Model

## Overview

The LSTM model predicts the **direction** of next-day carbon price movements (up vs down/flat) rather than exact returns. This binary classification approach addresses the fundamental challenge of financial time series: predicting direction is often more practical and achievable than predicting precise magnitudes.

**Key Achievements:**
- Successfully handles severe class imbalance (70% down/flat, 30% up)
- Achieves 63% recall on minority class (up movements)
- Prevents lazy solution of always predicting majority class
- Provides actionable trading signals with balanced precision/recall trade-off

## Design Decisions

### 1. Classification Over Regression

**Initial Approach:** Predict log returns (continuous values)
- **Problem:** Low R² (-0.004), poor directional accuracy (31%)
- **Issue:** Model struggled with noisy return magnitudes

**Solution:** Binary direction classification
- **Rationale:** Trading decisions primarily need direction, not exact magnitude
- **Result:** Clearer objective, better-defined success metrics

### 2. Binary vs Multi-Class

**Decision:** Two classes (Down/Flat vs Up)
- **Down/Flat (0):** log_return ≤ 0
- **Up (1):** log_return > 0

**Why not three classes (Down, Flat, Up)?**
- Carbon markets have many zero-return days (no trading)
- "Flat" is economically similar to "Down" for trading
- Binary simplifies the problem while maintaining practical value

### 3. Class Imbalance Handling

**Problem:** Severe imbalance leading to lazy predictions
```
Training Distribution:
  Down/Flat: 72% (1,705 samples)
  Up:        28% (661 samples)
```

**Solution:** BCEWithLogitsLoss with pos_weight
```python
pos_weight = num_negatives / num_positives  # ≈ 2.54
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Impact:**
- Unbalanced model: 38% recall (misses 62% of up movements)
- Balanced model: 63% recall (catches most opportunities)

## Architecture

### Model Structure

```
Input (60 days × 51 features)
    ↓
LSTM Layer 1 (128 hidden units)
    ↓
LSTM Layer 2 (128 hidden units)
    ↓
Batch Normalization
    ↓
FC Layer 1 (128 → 64)
    ↓
FC Layer 2 (64 → 32)
    ↓
Output Layer (32 → 1)
    ↓
Sigmoid → Binary Prediction
```

### Key Components

**LSTM Layers:**
- 2-layer stacked LSTM with 128 hidden units
- Dropout (0.2) between layers for regularization
- Captures temporal dependencies in 60-day windows

**Fully Connected Layers:**
- Progressive dimension reduction (128 → 64 → 32 → 1)
- ReLU activation and dropout between layers
- Final layer outputs raw logit for BCEWithLogitsLoss

**Input Specifications:**
- Sequence length: 60 trading days
- Features: 51 (prices, volumes, technical indicators, macro data)
- Dynamic input size detection from data

## Training Strategy

### Data Splits

Chronological splitting to prevent look-ahead bias:

| Split | Date Range | Purpose | Samples |
|-------|------------|---------|---------|
| Train | 2014 → 2020 | Model training | ~2,300 |
| Val | 2021 → 2022 | Hyperparameter tuning | ~670 |
| Test | 2023+ | Final evaluation | ~760 |

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
config.INPUT_SIZE = X_train.shape[2]  # Detects 51 features

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

## Performance Analysis

### Metrics Comparison

| Metric | Unbalanced | **Balanced** | Impact |
|--------|------------|--------------|--------|
| Accuracy | 63% | 61% | -2% (acceptable trade-off) |
| **Recall** | 38% | **63%** | +25% ✨ (catches more ups) |
| Precision | 38% | 40% | +2% (slightly better) |
| **F1-Score** | 38% | **49%** | +11% ✨ (better balance) |

### Trading Implications

**Balanced Model Advantages:**
1. **More Opportunities:** Identifies 63% of upward movements vs 38%
2. **Reduced Bias:** No longer defaults to always predicting down
3. **Actionable Signals:** Reasonable precision (40%) when predicting up

**Trade-offs:**
- Slightly lower overall accuracy (expected with balanced classes)
- More false positives (but also more true positives)
- Better suited for active trading strategies

### Confusion Matrix Analysis

```
Balanced Model Results:
                Predicted
              Down    Up
Actual Down   319    217  (Specificity: 60%)
       Up      83    142  (Sensitivity: 63%)
```

**Interpretation:**
- Model correctly identifies 63% of up movements (good recall)
- When predicting up, correct 40% of the time (moderate precision)
- Balanced approach prevents conservative bias

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
- `HIDDEN_SIZE`: 128 (LSTM hidden units)
- `NUM_LAYERS`: 2 (stacked LSTMs)
- `SEQUENCE_LENGTH`: 60 (days of history)
- `SHUFFLE_TRAIN_LOADER`: True (time series consideration)

## Conclusion

The LSTM direction prediction model successfully addresses the class imbalance problem inherent in carbon market data. By focusing on binary classification with appropriate class weighting, the model provides balanced predictions suitable for practical trading applications. The shift from regression to classification, combined with BCEWithLogitsLoss and pos_weight, transforms a biased predictor into a useful trading signal generator.

The key insight: **In financial markets, knowing direction with reasonable confidence is often more valuable than attempting to predict exact magnitudes with poor accuracy.**